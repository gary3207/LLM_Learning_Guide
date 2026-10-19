"""문서·예제 검사: 모델 로드·학습 없이 문법, 전문 일치, 파일 변환만 검사."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import tempfile


def data_value(source):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "data" for t in node.targets)]
    assert len(nodes) == 1
    return ast.literal_eval(nodes[0].value)


def main():
    root = Path(__file__).resolve().parents[2]
    first = root / "source/03-inference-and-data"
    second = root / "source/04-evaluation"
    examples = sorted(p for folder in (first, second) for p in folder.glob("*.py")
                        if p.name != Path(__file__).name)
    assert len(examples) == 9
    documents = [(root / name).read_text(encoding="utf-8")
                 for name in ("03-inference-and-data.md", "04-evaluation.md")]
    blocks = [code + "\n" for doc in documents
              for code in re.findall(r"(?ms)^```python\n(.*?)\n```$", doc)]
    assert len(blocks) == 9
    for path in examples:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert blocks.count(source) == 1, path.name
    augmented = (first / "train_classifier_augmented.py").read_text(encoding="utf-8")
    assert len(data_value(augmented)) == 10
    original = augmented.replace('    ("network unavailable", 1),\n', "", 1).replace(
        '/ "classifier_augmented.pt"', '/ "classifier.pt"', 1)
    assert len(data_value(original)) == 9

    # Only stdlib file generators run; examples training/inference never runs.
    scratch = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="classifier-doc-check-", dir=scratch) as temporary:
        try:
            os.chdir(temporary)
            Path("train_classifier.py").write_text(original, encoding="utf-8")
            for filename in ("classifier.pt", "classifier_augmented.pt"):
                Path(filename).write_bytes(b"sentinel-not-a-checkpoint")
            for script in (first / "make_augmented.py", second / "create_splits.py"):
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(script), run_name="__main__")
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(script), run_name="__main__")
                except FileExistsError:
                    pass
                else:
                    raise AssertionError(f"overwrite protection missing: {script.name}")
            assert Path("train_classifier.py").read_text(encoding="utf-8") == original
            assert Path("train_classifier_augmented.py").read_text(encoding="utf-8") == augmented
            dataset = json.loads(Path("classifier_splits.json").read_text(encoding="utf-8"))
            assert dataset["labels"] == ["GPU", "네트워크", "스토리지"]
            assert dataset["train"] == [list(item) for item in data_value(augmented)]
            assert {name: len(dataset[name]) for name in ("train", "validation", "test")} == {
                "train": 10, "validation": 4, "test": 3}
            assert dataset["validation"] == [["network error", 1], ["connection unavailable", 1],
                                             ["disk unavailable", 2], ["cuda error", 0]]
            assert dataset["test"] == [["cuda device error", 0],
                                      ["network connection unavailable", 1], ["disk space full", 2]]
            for filename in ("classifier.pt", "classifier_augmented.pt"):
                assert Path(filename).read_bytes() == b"sentinel-not-a-checkpoint"
        finally:
            os.chdir(previous)
    print("PASS: 9 examples scripts compile and match the full chapter code blocks.")
    print("PASS: augmentation round-trip, train 10 / validation 4 / test 3, exclusive writes.")
    print("NOT RUN: PyTorch model loading, inference, training, or VM commands.")


if __name__ == "__main__":
    main()
