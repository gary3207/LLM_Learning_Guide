"""Serve a verified selected registry version. Local, single-user demo only."""
import argparse
import importlib.util
from pathlib import Path


def dependency(name, relative):
    here = Path(__file__).resolve().parent
    path = here / (name + ".py")
    if not path.is_file():
        path = here.parent / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = dependency("smollm2_api_metrics", "21-api-serving/01-serve-qlora.py")
registry = dependency("smollm2_registry", "23-model-registry/01-model-registry.py")


def load_selected(store, device):
    record = registry.show(store)
    if record["base_model"] != api.MODEL or record["revision"] != api.REVISION:
        raise ValueError("selected base/revision is not supported by this QLoRA loader")
    manifest = registry.version_path(store, record["version"])
    digest = registry.fingerprint(manifest)["sha256"]
    tokenizer, model, count = api.load_model(Path(record["model_directory"]), device)
    # ponytail: trusted single writer; no file/selection changes during startup, not an atomic artifact snapshot.
    if registry.show(store) != record or registry.fingerprint(manifest)["sha256"] != digest:
        raise ValueError("selection changed during model loading; restart with stable inputs")
    identity = {"version": record["version"], "manifest_sha256": digest,
                "base_model": record["base_model"], "revision": record["revision"]}
    return tokenizer, model, count, identity


class Handler(api.Handler):
    def send_json(self, status, data):
        if status == 200:
            data = dict(data, model_version=self.server.identity)
        super().send_json(status, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be from 0 to 65535")
    tokenizer, model, count, identity = load_selected(args.registry, args.device)
    with api.HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.tokenizer, server.model, server.quantized_modules = tokenizer, model, count
        server.metrics, server.identity = args.metrics, identity
        print(f"READY http://127.0.0.1:{server.server_port} | device={model.device} | NF4={count}"
              f" | version={identity['version']}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("서버 종료", flush=True)


if __name__ == "__main__":
    main()
