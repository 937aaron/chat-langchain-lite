import re
import unittest

from agent.tools import get_security_advice, get_setup_guide, lookup_concept

STALE_DOCS_DOMAINS = ("python.langchain.com", "js.langchain.com")


def _concept_min_python(concept: str) -> str:
    """Return the minimum Python version reported by lookup_concept."""
    output = lookup_concept.invoke({"concept_name": concept})
    match = re.search(r"Minimum Python: (\d+\.\d+)\+", output)
    assert match, output
    return match.group(1)


def _guide_min_python(package: str) -> str:
    """Return the minimum Python version the installation guide gives for a package."""
    output = get_setup_guide.invoke({"topic": "installation"})
    match = re.search(rf"(\d+\.\d+) for {package}\b", output)
    assert match, output
    return match.group(1)


def _security_advice() -> str:
    """Return the security advice tool output."""
    return get_security_advice.invoke({"query": "documentation links"})


class LangGraphMinPythonTest(unittest.TestCase):
    def test_concept_lookup_reports_3_10(self):
        output = lookup_concept.invoke({"concept_name": "langgraph"})
        self.assertIn("Minimum Python: 3.10+", output)
        self.assertNotIn("3.7", output)

    def test_installation_guide_reports_3_10(self):
        self.assertEqual(_guide_min_python("langgraph"), "3.10")

    def test_concept_lookup_matches_installation_guide(self):
        for package in ("langchain", "langgraph", "langsmith"):
            with self.subTest(package=package):
                self.assertEqual(
                    _concept_min_python(package), _guide_min_python(package)
                )


class DocsDomainTest(unittest.TestCase):
    def test_security_advice_names_docs_langchain_com_as_canonical(self):
        output = _security_advice()
        self.assertIn("link users to docs.langchain.com", output)

    def test_security_advice_does_not_recommend_stale_domains(self):
        for line in _security_advice().splitlines():
            if "canonical" in line or "link users to" in line:
                for domain in STALE_DOCS_DOMAINS:
                    with self.subTest(line=line, domain=domain):
                        self.assertNotIn(f"link users to {domain}", line)
                        if domain in line:
                            self.assertIn("legacy", line)

    def test_all_docs_references_use_the_same_domain(self):
        output = _security_advice()
        domains = set(re.findall(r"\b[\w.]*langchain\.com\b", output))
        self.assertEqual(domains - set(STALE_DOCS_DOMAINS), {"docs.langchain.com"})


if __name__ == "__main__":
    unittest.main()
