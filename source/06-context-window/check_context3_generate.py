"""Exercise actual input guards without loading PyTorch or a checkpoint."""
import ast
from pathlib import Path

path = Path(__file__).with_name("06-generate-context3.py")
assert path.is_file(), "세 토큰 생성 코드가 아직 없습니다"
tree = ast.parse(path.read_text(encoding="utf-8"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "generate")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
for start, limit in [("unknown", 10), ("<bos>", 10), ("<eos>", 10),
                     ([], 10), ("gpu", 0), ("gpu", -1), ("gpu", True), ("gpu", 1.5)]:
    try:
        namespace["generate"](start, limit)
    except ValueError:
        pass
    else:
        raise AssertionError(f"잘못된 입력을 허용했습니다: {start!r}, {limit!r}")
print("PASS: 생성 함수의 잘못된 시작 토큰·길이 입력 거부 (모델 추론 미실행)")
