"""Offline guide checks; parses Python but NEVER runs embedded commands."""
from pathlib import Path
import argparse
import ast
import json
import re
from urllib.parse import unquote


# Explicit reviewed bindings: runtime filenames differ from their linked sources.
# ponytail: new full-script copies need a binding; never infer identity from similarity.
CANONICAL_SCRIPTS = [
    ("11-checkpoint-generation.md", "Python heredoc tiny_transformer_checkpoint.py", "source/09-transformer/03-checkpoint-generate.py"),
    ("15-finish-tiny-model.md", "Python heredoc tiny_transformer_finish.py", "source/09-transformer/07-finish-model.py"),
    ("16-pretrained-inference.md", "Python heredoc pretrained_inference.py", "source/16-pretrained/01-tokenize-generate.py"),
    ("17-sft-compare.md", "Python heredoc smollm2_sft.py", "source/17-sft/01-sft-compare.py"),
    ("18-lora-gpu.md", "Python heredoc smollm2_lora.py", "source/18-lora/01-lora-gpu.py"),
    ("19-model-evaluation.md", "Python heredoc smollm2_evaluate.py", "source/19-model-evaluation/01-evaluate-models.py"),
    ("20-qlora.md", "Python heredoc smollm2_qlora.py", "source/20-qlora/01-qlora.py"),
    ("21-api-serving.md", "Python heredoc smollm2_api.py", "source/21-api-serving/01-serve-qlora.py"),
    ("22-serving-benchmark.md", "Python heredoc smollm2_api_metrics.py", "source/21-api-serving/01-serve-qlora.py"),
    ("22-serving-benchmark.md", "Python heredoc smollm2_benchmark.py", "source/21-api-serving/02-benchmark.py"),
    ("23-model-registry.md", "Python heredoc smollm2_registry.py", "source/23-model-registry/01-model-registry.py"),
    ("24-registered-serving.md", "generated code", "source/24-registered-serving/01-serve-selected.py"),
    ("25-quality-gate.md", "generated code", "source/25-quality-gate/01-quality-gate.py"),
    ("26-run-tracking.md", "generated code", "source/26-run-tracking/01-track-run.py"),
    ("27-evaluation-pipeline.md", "generated code", "source/27-evaluation-pipeline/01-pipeline.py"),
    ("28-deployment-rollback.md", "generated smollm2_deploy.py", "source/28-deployment-rollback/01-deploy-rollback.py"),
    ("29-operations-load.md", "generated code", "source/29-operations-load/01-observe-load.py"),
    ("31-evaluation-data.md", "Python heredoc prepare_evaluation_data.py", "source/31-evaluation-data/01-prepare-data.py"),
    ("32-quality-improvement.md", "Python heredoc improve_qlora.py", "source/32-quality-improvement/01-improve-qlora.py"),
    ("33-training-resume.md", "Python heredoc resume_qlora.py", "source/33-training-resume/01-resume-qlora.py"),
    ("34-structure-comparison.md", "generated code", "source/34-structure-comparison/01-compare-structure.py"),
]


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


def python_heredocs(code):
    """Extract Python stdin/cat-to-.py bodies; never invoke a shell."""
    # ponytail: known guide wrappers, not a Bash parser; extend for new wrapper forms.
    lines = iter(code.splitlines())
    for line in lines:
        while line.endswith("\\"):
            line = line[:-1] + next(lines, "")
        match = re.search(r"<<\s*(['\"]?)(\w+)\1", line)
        if not match:
            continue
        command = line[:match.start()]
        python = (re.search(r"\bpython(?:3(?:\.\d+)?)?\s+-\s*$", command) or
                  re.search(r"--entrypoint\s+python(?:3(?:\.\d+)?)?\s+.*\s-\s*$", command))
        target = re.search(r"\bcat\s*>\s*(['\"]?)([\w./-]+\.py)\1\s*$", command)
        shell = re.search(r"\b(?:bash|sh)\s*$", command)
        body = []
        for line in lines:
            if line == match[2]:
                if python or target:
                    yield target[2] if target else None, "\n".join(body)
                elif shell:
                    yield from python_heredocs("\n".join(body))
                break
            body.append(line)
        else:
            if python or target or shell:
                raise ValueError(f"unclosed Python heredoc {match[2]}")


def check(root):
    errors = []
    documents = sorted(root.glob("*.md")) + sorted((root / "templates").glob("*.md"))
    scripts = [root / "source/check_guide.py"] + sorted((root / "source").glob("test_check_guide.py")) + sorted(
        path for directory in ("01-foundations", "02-first-classifier", "03-inference-and-data",
                               "04-evaluation", "05-next-token", "06-context-window", "08-attention",
                               "09-transformer", "16-pretrained", "17-sft", "18-lora", "19-model-evaluation", "20-qlora", "21-api-serving", "23-model-registry", "24-registered-serving", "25-quality-gate", "26-run-tracking", "27-evaluation-pipeline", "28-deployment-rollback", "29-operations-load", "30-container-runtime", "31-evaluation-data", "32-quality-improvement", "33-training-resume", "34-structure-comparison")
        for path in (root / "source" / directory).glob("*.py")
    )
    block_count = 0
    link_count = 0
    heredoc_count = 0
    generated_count = 0
    canonical_count = 0
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
        examples = []
        for language, code in blocks:
            if language in ("python", "py"):
                block_count += 1
                examples.append(("Python block", code))
            elif language in ("bash", "sh", "shell"):
                try:
                    bodies = list(python_heredocs(code))
                    heredoc_count += len(bodies)
                    examples.extend((f"Python heredoc {target or 'stdin'}", body) for target, body in bodies)
                except ValueError as exc:
                    errors.append(f"{name}: {exc}")
        embedded = {}
        for label, code in examples:
            try:
                tree = ast.parse(code, filename=name)
            except SyntaxError as exc:
                errors.append(f"{name} {label}: {exc}")
                continue
            embedded.setdefault(label, []).append(tree)
            # Only top-level literal code/files assignments, never evaluate generators.
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
                values = []
                if "code" in targets:
                    values.append(("code", node.value))
                if "files" in targets and isinstance(node.value, ast.Dict):
                    values.extend((key.value, value) for key, value in zip(node.value.keys, node.value.values)
                                  if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.endswith(".py"))
                for filename, value in values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        examples.append((f"generated {filename}", value.value))
                        generated_count += 1
        # Ignore fenced examples when checking navigational links.
        stripped = re.sub(r"(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text)
        targets = re.findall(r"\[[^\]\n]+\]\(([^)\s]+)\)", stripped)
        for target in targets:
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            link_count += 1
            destination = (path.parent / unquote(target.split("#")[0])).resolve()
            if not destination.is_relative_to(root.resolve()) or not destination.exists():
                errors.append(f"{name}: broken/outside link {target}")
        for document, label, source in CANONICAL_SCRIPTS:
            if document != name:
                continue
            copies = embedded.get(label, [])
            if source not in targets or len(copies) != 1:
                errors.append(f"{name}: canonical binding requires one {label} and link {source}")
                continue
            try:
                canonical = ast.parse((root / source).read_text(encoding="utf-8"), filename=source)
            except (OSError, SyntaxError) as exc:
                errors.append(f"{name}: canonical source {source}: {exc}")
                continue
            canonical_count += 1
            if ast.dump(copies[0]) != ast.dump(canonical):
                errors.append(f"{name} {label}: canonical drift from {source}")
    required = ["README.md", "00-roadmap.md", "01-foundations.md", "02-first-classifier.md", "03-inference-and-data.md", "04-evaluation.md", "05-next-token.md", "06-context-window.md", "07-resume.md", "CONTRIBUTING.md", "CHANGELOG.md", "templates/lesson.md"]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    return {"documents": len(documents), "python_files": len(scripts), "python_blocks": block_count, "local_links": link_count, "errors": errors, "training_executed": False,
            "python_heredocs": heredoc_count, "generated_python_scripts": generated_count,
            "canonical_comparisons": canonical_count}


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
