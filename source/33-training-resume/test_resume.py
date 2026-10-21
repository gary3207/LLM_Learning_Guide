"""Small real-PyTorch check for full-state restoration and omission failures."""
import copy
import os
from pathlib import Path
import runpy
import tempfile

SCRIPT = Path(__file__).with_name("01-resume-qlora.py")


def check():
    assert SCRIPT.is_file(), "training-resume script missing"
    import torch

    module = runpy.run_path(str(SCRIPT))
    torch.set_num_threads(1)

    def setup():
        params = {"weight": torch.nn.Parameter(torch.tensor([0.3, -0.2]))}
        optimizer = torch.optim.AdamW(params.values(), lr=0.001, weight_decay=0.0,
                                     foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        return params, optimizer, scheduler

    def advance(params, optimizer, scheduler, start, end, trace):
        for step in range(start, end):
            indices = torch.randperm(8)[:2].tolist()
            optimizer.zero_grad(set_to_none=True)
            x = torch.tensor([sum(indices) / 10, indices[0] + 1.0])
            loss = ((params["weight"] * x - torch.tensor([1.0, -1.0])) ** 2).mean()
            lr = optimizer.param_groups[0]["lr"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            optimizer.step()
            scheduler.step()
            trace.append({"step": step + 1, "batch": indices, "lr": lr, "loss": loss.item()})

    params, optimizer, scheduler = setup()
    torch.manual_seed(1234)
    initial = module["capture"](params, optimizer, scheduler, 0, [])
    trace = []
    advance(params, optimizer, scheduler, 0, 6, trace)
    reference = module["capture"](params, optimizer, scheduler, 6, trace)
    module["restore"](params, optimizer, scheduler, initial)
    trace = []
    advance(params, optimizer, scheduler, 0, 3, trace)
    checkpoint = module["capture"](params, optimizer, scheduler, 3, trace)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        path = Path(tmp) / "state.pt"
        with path.open("xb") as stream:
            torch.save(checkpoint, stream)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for fault in (None, "optimizer", "scheduler", "rng"):
        params, optimizer, scheduler = setup()
        torch.manual_seed(999)  # Model/process setup need not preserve the training RNG.
        module["restore"](params, optimizer, scheduler, checkpoint)
        if fault == "optimizer":
            optimizer.state.clear()
        elif fault == "scheduler":
            scheduler.load_state_dict(initial["scheduler"])
        elif fault == "rng":
            torch.manual_seed(999)
        trace = copy.deepcopy(checkpoint["trace"])
        advance(params, optimizer, scheduler, 3, 6, trace)
        resumed = module["capture"](params, optimizer, scheduler, 6, trace)
        try:
            result = module["compare"](reference, resumed)
        except (AssertionError, ValueError):
            assert fault is not None, "correct restoration failed"
        else:
            assert fault is None, f"missing {fault} state escaped comparison"
            assert result["max_adapter_abs_diff"] == 0.0
    print("PASS real-tensor full-state replay; optimizer/scheduler/RNG omission each rejected")


if __name__ == "__main__":
    check()
