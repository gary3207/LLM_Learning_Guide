"""Stdlib filesystem/CLI/static checks; these do not build or run Docker."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
APPS = {
    "smollm2_serve_selected.py": "24-registered-serving/01-serve-selected.py",
    "smollm2_api_metrics.py": "21-api-serving/01-serve-qlora.py",
    "smollm2_registry.py": "23-model-registry/01-model-registry.py",
}
RECIPES = ("Dockerfile", "requirements.txt", ".dockerignore")


class ContainerTests(unittest.TestCase):
    def run_helper(self, source, output):
        return subprocess.run([sys.executable, "-B", str(HERE / "01-prepare-context.py"),
                               "--source", str(source), "--output", str(output)],
                              capture_output=True, text=True)

    def test_repository_context_bytes(self):
        source = HERE.parent
        before = {name: (source / relative).read_bytes() for name, relative in APPS.items()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "context"
            result = self.run_helper(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {**{f"app/{name}": data for name, data in before.items()},
                        **{name: (HERE / name).read_bytes() for name in RECIPES}}
            self.assertEqual({p.relative_to(output).as_posix(): p.read_bytes()
                              for p in output.rglob("*") if p.is_file()}, expected)
            self.assertIn("FILES 6", result.stdout)
            for name, data in before.items():
                self.assertIn(hashlib.sha256(data).hexdigest() + "  app/" + name, result.stdout)
                self.assertEqual((source / APPS[name]).read_bytes(), data)

    def test_flat_vm_context_excludes_other_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            expected = {}
            for name, relative in APPS.items():
                expected[name] = (HERE.parent / relative).read_bytes()
                (source / name).write_bytes(expected[name])
            for name in ("model.safetensors", "cache", "credentials"):
                (source / name).write_bytes(b"not a context input")
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            output = root / "context"
            result = self.run_helper(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({p.name: p.read_bytes() for p in (output / "app").iterdir()}, expected)
            self.assertEqual(len([p for p in output.rglob("*") if p.is_file()]), 6)
            self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, before)

    def test_bad_app_preflight_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name, relative in APPS.items():
                (source / name).write_bytes((HERE.parent / relative).read_bytes())
            last = source / "smollm2_registry.py"
            for bad in (b"def invalid(:", None):
                with self.subTest(bad=bad):
                    if bad is None:
                        last.unlink()
                    else:
                        last.write_bytes(bad)
                    before = {p.name: p.read_bytes() for p in source.iterdir()}
                    output = root / ("missing-context" if bad is None else "invalid-context")
                    result = self.run_helper(source, output)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())
                    self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, before)

    def test_symlink_input_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name, relative in APPS.items():
                (source / name).write_bytes((HERE.parent / relative).read_bytes())
            victim = source / "smollm2_registry.py"
            victim.unlink()
            secret = root / "secret.py"
            secret.write_bytes(b"PRIVATE = 1\n")
            try:
                victim.symlink_to(secret)
            except OSError as exc:
                self.skipTest("symlink creation unavailable: " + str(exc))
            for dangling in (False, True):
                if dangling:
                    secret.unlink()
                output = root / ("dangling-context" if dangling else "linked-context")
                result = self.run_helper(source, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                self.assertTrue(victim.is_symlink())
                if not dangling:
                    self.assertEqual(secret.read_bytes(), b"PRIVATE = 1\n")

    def test_existing_outputs_are_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (False, True):
                output = root / ("directory" if directory else "file")
                if directory:
                    output.mkdir()
                    (output / "keep").write_bytes(b"keep")
                else:
                    output.write_bytes(b"keep")
                result = self.run_helper(HERE.parent, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((output / "keep" if directory else output).read_bytes(), b"keep")
                if directory:
                    self.assertEqual([p.name for p in output.iterdir()], ["keep"])
            missing = root / "absent" / "context"
            self.assertNotEqual(self.run_helper(HERE.parent, missing).returncode, 0)
            self.assertFalse(missing.parent.exists())

    def test_output_symlinks_are_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            (target / "keep").write_bytes(b"keep")
            output = root / "context"
            try:
                output.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest("symlink creation unavailable: " + str(exc))
            self.assertNotEqual(self.run_helper(HERE.parent, output).returncode, 0)
            self.assertEqual([p.name for p in target.iterdir()], ["keep"])
            self.assertEqual((target / "keep").read_bytes(), b"keep")
            (target / "keep").unlink()
            target.rmdir()
            self.assertNotEqual(self.run_helper(HERE.parent, output).returncode, 0)
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())

    def test_recipe_contract(self):
        docker = (HERE / "Dockerfile").read_text()
        self.assertTrue(docker.startswith("FROM python:3.12-slim-bookworm@sha256:5d9e6adc9abd558bc0ce05f9ab16cef71328916fc9728f6b81e337af86a737e2\n"))
        self.assertEqual((HERE / "requirements.txt").read_text().splitlines(), [
            "torch==2.11.0+cu128", "transformers==4.57.1", "peft==0.17.1",
            "bitsandbytes==0.48.2", "accelerate==1.15.0", "huggingface_hub==0.36.2"])
        for token in ("--no-install-recommends libgomp1 libstdc++6", "HF_HUB_OFFLINE=1",
                      "TRANSFORMERS_OFFLINE=1", "HF_HUB_CACHE=/cache/hub", "HOME=/tmp",
                      "USER=llm", "LOGNAME=llm", "USER 65532:65532",
                      "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1",
                      "PIP_NO_CACHE_DIR=1", "PIP_DISABLE_PIP_VERSION_CHECK=1",
                      "--index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'",
                      "python -m pip check", "python -m pip freeze > /opt/llm/image-packages.txt"):
            self.assertIn(token, docker)
        self.assertLess(docker.index("WORKDIR /opt/llm"), docker.index("image-packages.txt"))
        self.assertLess(docker.index("download.pytorch.org"), docker.index("pip install -r requirements.txt"))
        self.assertEqual([line for line in docker.splitlines() if line.startswith("COPY ")],
                         ["COPY requirements.txt ./"] + [f"COPY app/{name} ./" for name in APPS])
        self.assertIn('ENTRYPOINT ["python", "-B", "smollm2_serve_selected.py"]', docker)
        self.assertIn('CMD ["--device", "cuda", "--metrics", "--port", "8003"]', docker)
        self.assertEqual((HERE / ".dockerignore").read_text().splitlines(),
                         ["**", "!Dockerfile", "!requirements.txt", "!app/", *[f"!app/{name}" for name in APPS]])


if __name__ == "__main__":
    unittest.main()
