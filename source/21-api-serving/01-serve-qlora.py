"""Saved QLoRA HTTP demo. Loopback only; not a production server."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import traceback

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def parse_request(body):
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict) or set(data) - {"prompt", "max_new_tokens"}:
        raise ValueError("Use a JSON object with prompt and optional max_new_tokens")
    prompt, limit = data.get("prompt"), data.get("max_new_tokens", 8)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512:
        raise ValueError("prompt must be a nonblank string of at most 512 characters")
    prompt.encode("utf-8")  # Reject unpaired JSON surrogates before tokenization.
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("max_new_tokens must be an integer from 1 to 32")
    return prompt, limit


def load_model(directory, device):
    import bitsandbytes as bnb
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가: CPU로 자동 전환하지 않습니다.")
    settings = json.loads((directory / "qlora_settings.json").read_text(encoding="utf-8"))
    lora = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"], "bias": "none"}
    expected = {"base_model": MODEL, "revision": REVISION, "dtype": "float32", "attention": "eager",
                "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                 "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float32"},
                "lora": lora}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("QLoRA settings mismatch")
    adapter = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    adapter["target_modules"] = sorted(adapter.get("target_modules", []))
    expected_adapter = dict(lora, base_model_name_or_path=MODEL, revision=REVISION,
                            peft_type="LORA", task_type="CAUSAL_LM")
    if any(adapter.get(key) != value for key, value in expected_adapter.items()):
        raise ValueError("adapter config mismatch")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation="eager", device_map={"": device},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, directory, is_trainable=False, local_files_only=True).eval()
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert quantized and all(m.weight.quant_state is not None and m.weight.quant_state.quant_type == "nf4"
                             and m.weight.quant_state.nested for m in quantized)
    assert all(p.device.type == device for p in model.parameters()), "자동 offload 금지"
    assert not any(p.requires_grad for p in model.parameters()), "추론 전용 모델이어야 합니다"
    return tokenizer, model, len(quantized)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self.send_json(404, {"error": "not found"})
        self.send_json(200, {"status": "ok", "device": str(self.server.model.device),
                             "quantized_modules": self.server.quantized_modules})

    def do_POST(self):
        if self.path != "/generate":
            return self.send_json(404, {"error": "not found"})
        if self.headers.get_content_type() != "application/json":
            return self.send_json(415, {"error": "Content-Type must be application/json"})
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            return self.send_json(400, {"error": "one Content-Length and no Transfer-Encoding required"})
        try:
            length = int(lengths[0])
        except ValueError:
            return self.send_json(400, {"error": "invalid Content-Length"})
        if not 1 <= length <= 4096:
            return self.send_json(413, {"error": "body must contain 1 to 4096 bytes"})
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("incomplete request body")
            prompt, limit = parse_request(body)
        except (ValueError, TimeoutError, RecursionError) as exc:
            return self.send_json(400, {"error": str(exc)})
        import torch
        from transformers import GenerationConfig

        tokenizer, model = self.server.tokenizer, self.server.model
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=True, add_generation_prompt=True)
        if len(prefix) > 256:
            return self.send_json(400, {"error": "chat-formatted input exceeds 256 tokens"})
        try:
            inputs = torch.tensor([prefix], device=model.device)
            config = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1, use_cache=True,
                repetition_penalty=1.0, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            with torch.inference_mode():
                generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=config)
            result = {"text": tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True),
                      "input_tokens": inputs.shape[1], "generated_tokens": generated.shape[1] - inputs.shape[1]}
        except Exception:
            traceback.print_exc()
            return self.send_json(500, {"error": "inference failed; check server terminal"})
        self.send_json(200, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", nargs="?", type=Path, default=Path("../smollm2_qlora_v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be from 0 to 65535")
    tokenizer, model, count = load_model(args.adapter, args.device)
    # ponytail: loopback + one request at a time; production needs a serving stack, not external exposure of this demo.
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.tokenizer, server.model, server.quantized_modules = tokenizer, model, count
        print(f"READY http://127.0.0.1:{server.server_port} | device={model.device} | NF4={count}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("서버 종료", flush=True)
