"""Offline guide checks; compiles code but NEVER runs training/VM commands."""
from pathlib import Path
import argparse
import ast
import json
import re
from urllib.parse import unquote


def fences(text):
    opened = None
    body = []
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})([^`~]*)$", line)
        if opened is None and match:
            opened = (match[1], match[2].strip().split(" ")[0])
            body = []
        elif opened and re.match(r"^\s*" + re.escape(opened[0][0]) + "{" + str(len(opened[0])) + r",}\s*$", line):
            yield opened[1], "\n".join(body)
            opened = None
        elif opened:
            body.append(line)
    if opened:
        raise ValueError("unclosed code fence")


def check(root):
    errors = []
    documents = sorted(root.glob("*.md")) + sorted((root / "templates").glob("*.md"))
    scripts = [root / "source/check_guide.py"] + sorted(
        path for directory in ("01-foundations", "02-first-classifier", "03-inference-and-data",
                               "04-evaluation", "05-next-token", "06-context-window", "08-attention",
                               "09-transformer", "16-pretrained", "17-sft", "18-lora", "19-model-evaluation", "20-qlora", "21-api-serving", "23-model-registry", "24-registered-serving", "25-quality-gate", "26-run-tracking", "27-evaluation-pipeline", "28-deployment-rollback")
        for path in (root / "source" / directory).glob("*.py")
    )
    block_count = 0
    link_count = 0
    secret = re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}")
    for path in documents + scripts:
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(root).as_posix()
        if secret.search(text):
            errors.append(f"{name}: possible credential")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                errors.append(f"{name}: {exc}")
            continue
        try:
            blocks = list(fences(text))
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        for language, code in blocks:
            if language in ("python", "py"):
                block_count += 1
                try:
                    ast.parse(code, filename=name)
                except SyntaxError as exc:
                    errors.append(f"{name} Python block: {exc}")
        # Ignore fenced examples when checking navigational links.
        stripped = re.sub(r"(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text)
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\s]+)\)", stripped):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            link_count += 1
            destination = (path.parent / unquote(target.split("#")[0])).resolve()
            if not destination.is_relative_to(root.resolve()) or not destination.exists():
                errors.append(f"{name}: broken/outside link {target}")
    required = ["README.md", "00-roadmap.md", "01-foundations.md", "02-first-classifier.md", "03-inference-and-data.md", "04-evaluation.md", "05-next-token.md", "06-context-window.md", "07-resume.md", "CONTRIBUTING.md", "CHANGELOG.md", "templates/lesson.md"]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    return {"documents": len(documents), "python_files": len(scripts), "python_blocks": block_count, "local_links": link_count, "errors": errors, "training_executed": False}


def self_check():
    assert list(fences("```python\nx = 1\n```")) == [("python", "x = 1")]
    assert list(fences("````python\nx = 2\n````")) == [("python", "x = 2")]
    try:
        list(fences("```python\nx = 1"))
    except ValueError:
        pass
    else:
        raise AssertionError("unclosed fence was accepted")


if __name__ == "__main__":
    self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["errors"]))
