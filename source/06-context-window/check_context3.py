"""Check actual training-data preparation without importing PyTorch or training."""
import ast
from pathlib import Path

path = Path(__file__).with_name("05-train-context3.py")
assert path.is_file(), "세 토큰 학습 코드가 아직 없습니다"
tree = ast.parse(path.read_text(encoding="utf-8"))
start = next(i for i, node in enumerate(tree.body)
             if isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id == "sentences" for t in node.targets))
stop = next(i for i, node in enumerate(tree.body)
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "x" for t in node.targets))
namespace = {}
exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]), str(path), "exec"), namespace)
assert namespace["pairs"] == [
    (["<bos>", "<bos>", "gpu"], "current"),
    (["<bos>", "gpu", "current"], "status"),
    (["gpu", "current", "status"], "failed"),
    (["current", "status", "failed"], "<eos>"),
    (["<bos>", "<bos>", "network"], "current"),
    (["<bos>", "network", "current"], "status"),
    (["network", "current", "status"], "healthy"),
    (["current", "status", "healthy"], "<eos>"),
]
assert len(namespace["vocab"]) == 8
print("PASS: 실제 데이터 준비에서 8쌍·3토큰·문장 경계·어휘 8개 확인 (학습 미실행)")
