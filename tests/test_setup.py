import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import setup

README = Path(__file__).resolve().parents[1] / "README.md"


def _feedback(key, score=None, extra=None, comment=None):
    return SimpleNamespace(key=key, score=score, value=None, extra=extra, comment=comment)


class JudgeModelJsonTest(unittest.TestCase):
    def test_routes_through_gateway_without_raw_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "lsv2_sk_raw_secret", "ANTHROPIC_BASE_URL": "https://api.anthropic.com"}):
            model_json = setup.judge_model_json()

        kwargs = model_json["kwargs"]
        self.assertEqual(kwargs["anthropic_api_url"], setup.GATEWAY_ANTHROPIC_BASE_URL)
        self.assertEqual(kwargs["model"], setup.JUDGE_MODEL)
        self.assertNotIn("lsv2_sk_raw_secret", json.dumps(model_json))

    def test_key_is_workspace_secret_reference(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "lsv2_sk_raw_secret"}):
            model_json = setup.judge_model_json()

        self.assertEqual(
            model_json["kwargs"]["anthropic_api_key"],
            {"lc": 1, "type": "secret", "id": ["ANTHROPIC_API_KEY"]},
        )


class VerifyOnlineEvaluatorsTest(unittest.TestCase):
    def _run(self, feedback):
        client = mock.Mock()
        client.list_runs.return_value = [SimpleNamespace(id="run-1")]
        client.list_feedback.return_value = feedback
        fake_agent = types.ModuleType("agent.agent")
        fake_agent.invoke_agent = mock.Mock()
        with mock.patch.dict(sys.modules, {"agent.agent": fake_agent}), \
                mock.patch("langsmith.Client", return_value=client), \
                mock.patch.object(setup.time, "sleep"):
            return setup.verify_online_evaluators(timeout_s=60, poll_s=1)

    def test_reports_evaluator_errors(self):
        keys = [ev["feedback_key"] for ev in setup.EVALUATORS]
        feedback = [_feedback(k, extra={"error": True}, comment="Anthropic authentication failed") for k in keys]

        errors = self._run(feedback)

        self.assertEqual(len(errors), len(keys))
        self.assertIn("Anthropic authentication failed", errors[0])

    def test_success_returns_no_errors(self):
        feedback = [_feedback(ev["feedback_key"], score=1) for ev in setup.EVALUATORS]

        self.assertEqual(self._run(feedback), [])


class MainTest(unittest.TestCase):
    def _run_main(self, evaluator_errors):
        fake_hub = types.ModuleType("utils.context_hub")
        fake_hub.push_agents_md = mock.Mock()
        fake_hub.push_demo_skills = mock.Mock()
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"utils.context_hub": fake_hub}), \
                mock.patch.dict(os.environ, {"LANGSMITH_API_KEY": "lsv2_pt_test"}), \
                mock.patch.object(sys, "argv", ["setup", "--skip-baseline-experiments"]), \
                mock.patch.object(setup, "ensure_project_exists"), \
                mock.patch.object(setup, "setup_dataset"), \
                mock.patch.object(setup, "setup_online_evaluators", return_value=["rule-1"]), \
                mock.patch.object(setup, "verify_online_evaluators", return_value=evaluator_errors), \
                mock.patch("builtins.print") as printed:
            os.chdir(tmp)
            try:
                setup.main()
            finally:
                os.chdir(cwd)
        return " ".join(str(c.args) for c in printed.call_args_list)

    def test_exits_nonzero_when_evaluators_error(self):
        with self.assertRaises(SystemExit) as exc:
            self._run_main(["factual_accuracy: Anthropic authentication failed"])
        self.assertEqual(exc.exception.code, 1)

    def test_completes_when_evaluators_score(self):
        output = self._run_main([])
        self.assertIn("Setup complete.", output)


class ReadmeTest(unittest.TestCase):
    def test_documents_judge_secret_and_gateway(self):
        readme = README.read_text()
        self.assertIn(f"`{setup.JUDGE_API_KEY_SECRET}`", readme)
        self.assertIn(setup.GATEWAY_ANTHROPIC_BASE_URL, readme)
        self.assertIn("Settings → Secrets", readme)


if __name__ == "__main__":
    unittest.main()
