"""Read-only diagnostic quality gate; optionally serve the same approved in-memory model."""
import importlib.util
from importlib.metadata import version
import hashlib
import json
from pathlib import Path

GENERATION = {"max_new_tokens": 8, "do_sample": False, "num_beams": 1,
              "use_cache": True, "repetition_penalty": 1.0}

here = Path(__file__).resolve().parent
path = here / "smollm2_serve_selected.py"
if not path.is_file():
    path = here.parent / "24-registered-serving/01-serve-selected.py"
spec = importlib.util.spec_from_file_location("smollm2_serve_selected", path)
assert spec is not None and spec.loader is not None
selected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selected)
scorer = selected.dependency("smollm2_evaluate", "19-model-evaluation/01-evaluate-models.py")


def decide(records):
    if not isinstance(records, list) or len(records) != len(scorer.ROWS):
        raise ValueError("exactly the fixed eight rows are required")
    for row, expected in zip(records, scorer.ROWS):
        if (not isinstance(row, dict) or
                tuple(row.get(key) for key in ("group", "prompt", "answer")) != expected or
                not isinstance(row.get("text"), str)):
            raise ValueError("fixed row identity/order and generated text required")
    groups = {}
    for row in records:
        group = groups.setdefault(row["group"], {"correct": 0, "total": 0})
        group["total"] += 1
        group["correct"] += scorer.classify(row["text"], row["answer"]) == "correct"
    correct = sum(group["correct"] for group in groups.values())
    return {"correct": correct, "total": len(records), "groups": groups}, correct == 8


def evaluate(tokenizer, model):
    import torch
    from transformers import GenerationConfig

    config = GenerationConfig(**GENERATION, bos_token_id=tokenizer.bos_token_id,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    records = []
    with torch.inference_mode():
        for group, prompt, answer in scorer.ROWS:
            prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True)
            inputs = torch.tensor([prefix], device=model.device)
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=config)
            text = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
            records.append(dict(group=group, prompt=prompt, answer=answer, text=text,
                                input_tokens=inputs.shape[1], generated_tokens=generated.shape[1] - inputs.shape[1]))
            print(f"[{group}/{len(records)}] {scorer.classify(text, answer)} | answer={answer} | text={text!r}",
                  flush=True)
    return records


def run(store, device, output, serve=False, port=8000):
    store, output = Path(store), Path(output)
    if device not in ("cpu", "cuda") or type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("device must be cpu/cuda and port from 0 to 65535")
    if output.is_symlink() or output.exists():
        raise FileExistsError("quality report overwrite refused: " + str(output))
    record = selected.registry.show(store)
    target = output.resolve()
    if (target.is_relative_to(Path(record["model_directory"]).resolve()) or
            target.is_relative_to(store.resolve()) or target == Path(record["report_path"]).resolve()):
        raise ValueError("quality report must be outside model/registry and not the performance report")
    tokenizer, model, count, identity = selected.load_selected(store, device)
    records = evaluate(tokenizer, model)
    summary, approved = decide(records)
    report = {"schema": 1, "model_version": identity,
              "policy": {"criterion": "all 8 correct; pedagogical strict threshold, not a production criterion",
                         "scoring": "strip-only, case-sensitive exact match",
                         "dataset_sha256": hashlib.sha256(json.dumps(scorer.ROWS, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()},
              "runtime": dict(device=str(model.device), quantized_modules=count,
                              **{name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}),
              "generation": dict(GENERATION, bos_token_id=tokenizer.bos_token_id,
                                 eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id),
              "rows": records, "summary": summary, "approved": approved, "training_executed": False,
              "scope": "diagnostic DEV examples already seen; synthetic gpu=failed/network=healthy; not independent final test; no teacher-forcing CE; this launcher only"}
    # ponytail: trusted single local writer, not an atomic snapshot or concurrent-mutation guarantee.
    if (selected.registry.show(store) != record or
            identity["version"] != record["version"] or
            selected.registry.fingerprint(selected.registry.version_path(store, record["version"]))["sha256"]
            != identity["manifest_sha256"]):
        raise ValueError("selection/artifacts changed during quality gate; restart with stable inputs")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{'APPROVED' if approved else 'BLOCKED'} {summary['correct']}/{summary['total']}", flush=True)
    if approved and serve:
        with selected.api.HTTPServer(("127.0.0.1", port), selected.Handler) as server:
            server.tokenizer, server.model, server.quantized_modules = tokenizer, model, count
            server.identity, server.metrics = dict(identity), False
            print(f"READY http://127.0.0.1:{server.server_port} | device={model.device} | NF4={count}"
                  f" | version={identity['version']}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("Server stopped", flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("../model_registry_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("../qlora_quality_gate_v1.json"))
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        raise SystemExit(1 if exc.code else 0) from None
    if not 0 <= args.port <= 65535:
        parser.exit(1, "port must be from 0 to 65535\n")
    result = run(args.registry, args.device, args.output, args.serve, args.port)
    raise SystemExit(0 if result["approved"] else 2)
