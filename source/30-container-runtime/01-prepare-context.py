"""Copy only runtime recipes and existing app code into a NEW Docker context."""
import argparse
import ast
import hashlib
from pathlib import Path


def prepare(source, output):
    source, output = Path(source), Path(output)
    recipes = Path(__file__).resolve().parent
    apps = {
        "smollm2_serve_selected.py": "24-registered-serving/01-serve-selected.py",
        "smollm2_api_metrics.py": "21-api-serving/01-serve-qlora.py",
        "smollm2_registry.py": "23-model-registry/01-model-registry.py",
    }
    inputs = {name: recipes / name
              for name in ("Dockerfile", "requirements.txt", ".dockerignore")}
    for name, relative in apps.items():
        flat = source / name
        inputs[f"app/{name}"] = flat if flat.exists() or flat.is_symlink() else source / relative
    files = {}
    for name, path in inputs.items():
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("symbolic links are not supported: " + str(path))
        files[name] = path.read_bytes()
    for name, data in files.items():
        if name.startswith("app/"):
            ast.parse(data, filename=name)
    output.mkdir()
    (output / "app").mkdir()
    for name, data in files.items():
        (output / name).write_bytes(data)
    print("CONTEXT", output.absolute())
    print("FILES", len(files))
    for name, data in files.items():
        if name.startswith("app/"):
            print(hashlib.sha256(data).hexdigest() + "  " + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
