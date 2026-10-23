"""Stdlib-only offline regression checks; embedded commands are never run."""
import os
from pathlib import Path
import tempfile
import unittest

from check_guide import check, fences


class GuideCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("README.md", "00-roadmap.md", "01-foundations.md",
                     "02-first-classifier.md", "03-inference-and-data.md",
                     "04-evaluation.md", "05-next-token.md", "06-context-window.md",
                     "07-resume.md", "CONTRIBUTING.md", "CHANGELOG.md",
                     "templates/lesson.md", "source/check_guide.py"):
            self.write(name, "")

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def run_guide(self, text):
        self.write("README.md", text)
        return check(self.root)

    def test_direct_python_fences(self):
        result = self.run_guide("```python\nx = 1\n```\n```py\ny = 2\n```\n")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["python_blocks"], 2)
        self.assertFalse(result["training_executed"])
        self.assertTrue(self.run_guide("```python\nif :\n```\n")["errors"])

    def test_generated_code_literal(self):
        sentinel = self.root / "must-not-exist"
        for body in ("if :\n", "x = 1\n"):
            wrapper = (f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
                       f"code = {body!r}\nraise RuntimeError('never run')\n")
            result = self.run_guide("```bash\npython - <<'PY'\n" + wrapper + "PY\n```\n")
            self.assertFalse(sentinel.exists())
            if body.startswith("if"):
                self.assertTrue(result["errors"], result)
                self.assertIn("generated code", result["errors"][0])
            else:
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["generated_python_scripts"], 1)

    def test_generated_files_literals(self):
        for body in ("if :\n", "raise RuntimeError('never run')\n"):
            wrapper = f"files = {{'worker.py': {body!r}, 'unit.service': '[Unit] not Python'}}\n"
            result = self.run_guide("```python\n" + wrapper + "```\n")
            if body.startswith("if"):
                self.assertTrue(result["errors"], result)
                self.assertIn("generated worker.py", result["errors"][0])
            else:
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["generated_python_scripts"], 1)

    def test_cat_python_file_heredoc(self):
        for body in ("if :\n", "x = 1\n"):
            result = self.run_guide("```bash\n(set -C; cat > worker.py <<'PY'\n" + body + "PY\n)\n```\n")
            if body.startswith("if"):
                self.assertTrue(result["errors"], result)
                self.assertIn("worker.py", result["errors"][0])
            else:
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["python_heredocs"], 1)

    def test_container_python_heredoc(self):
        command = ("sudo -n docker run --rm -i \\\n"
                   "    --entrypoint python \\\n"
                   "    example:tag - <<'IMAGE_CHECK'\n")
        for body in ("if :\n", "raise RuntimeError('never run')\n"):
            result = self.run_guide("```bash\n" + command + body + "IMAGE_CHECK\n```\n")
            self.assertEqual(bool(result["errors"]), body.startswith("if"), result)
            self.assertEqual(result["python_heredocs"], 1)

    def test_explicit_linked_source_drift(self):
        source = "source/34-structure-comparison/01-compare-structure.py"
        helper = "source/34-structure-comparison/test_structure.py"
        self.write(source, "x = 1\n")
        self.write(helper, "raise RuntimeError('helper must not run')\n")
        for body in ("x = 2\n", "# formatting is not drift\nx=1\n"):
            self.write("34-structure-comparison.md",
                       f"[Code]({source}) [Helper]({helper})\n\n"
                       f"```bash\npython - <<'PY'\ncode = {body!r}\nPY\n```\n")
            result = check(self.root)
            if body.startswith("x = 2"):
                self.assertTrue(result["errors"], result)
                self.assertIn("canonical drift", result["errors"][0])
                self.assertIn(source, result["errors"][0])
            else:
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["canonical_comparisons"], 1)

    def test_checker_regression_file_is_scanned(self):
        self.write("source/test_check_guide.py", "if :\n")
        result = check(self.root)
        self.assertTrue(any("source/test_check_guide.py" in error for error in result["errors"]), result)

    def test_real_structure_guide_mutations(self):
        repository = Path(__file__).resolve().parent.parent
        source = "source/34-structure-comparison/01-compare-structure.py"
        self.write(source, (repository / source).read_text(encoding="utf-8"))
        document = (repository / "34-structure-comparison.md").read_text(encoding="utf-8")
        wrappers = [body for language, body in fences(document)
                    if language == "bash" and "code = r'''" in body]
        self.assertEqual(len(wrappers), 1)
        original = wrappers[0]
        needle = "STEPS, LR = 200, 0.1"
        self.assertEqual(original.count(needle), 1)
        for replacement, expected in ((needle, None),
                                      ("STEPS, LR = , 0.1", "generated code"),
                                      ("STEPS, LR = 201, 0.1", "canonical drift")):
            self.write("34-structure-comparison.md", f"[Code]({source})\n\n```bash\n"
                       + original.replace(needle, replacement) + "\n```\n")
            result = check(self.root)
            if expected:
                self.assertTrue(any(expected in error for error in result["errors"]), result)
            else:
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["canonical_comparisons"], 1)

    def test_missing_or_duplicate_canonical_body_fails(self):
        source = "source/34-structure-comparison/01-compare-structure.py"
        self.write(source, "x = 1\n")
        for body in ("x = 1", "code = 'x = 1'\ncode = 'x = 1'"):
            self.write("34-structure-comparison.md", f"[Code]({source})\n```python\n{body}\n```\n")
            result = check(self.root)
            self.assertTrue(any("canonical binding" in error for error in result["errors"]), result)

    def test_unclosed_heredoc_fails(self):
        result = self.run_guide("```bash\npython - <<'PY'\nx = 1\n```\n")
        self.assertTrue(any("unclosed Python heredoc" in error for error in result["errors"]), result)

    def test_nonpython_heredoc_is_not_executed_or_parsed(self):
        text = "```bash\ncat > note.txt <<'TEXT'\npython - <<'PY'\nif :\nPY\nTEXT\n```\n"
        result = self.run_guide(text)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["python_heredocs"], 0)

    def test_python_inside_bash_heredoc(self):
        for body in ("if :\n", "x = 1\n"):
            text = "```bash\nbash <<'CHECK'\npython - <<'REPORT'\n" + body + "REPORT\nCHECK\n```\n"
            result = self.run_guide(text)
            self.assertEqual(bool(result["errors"]), body.startswith("if"), result)
            self.assertEqual(result["python_heredocs"], 1)

    def test_python_heredoc_syntax(self):
        self.assertEqual(self.run_guide("```bash\npython - <<'PY'\nx = 1\nPY\n```\n")["errors"], [])
        result = self.run_guide("```bash\npython - <<'PY'\nif :\nPY\n```\n")
        self.assertTrue(result["errors"], result)
        self.assertIn("heredoc", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
