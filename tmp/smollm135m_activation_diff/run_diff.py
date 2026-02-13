#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_tokens_csv(tokens_csv: str) -> list[int]:
    tokens = []
    for raw in tokens_csv.split(","):
        value = raw.strip()
        if not value:
            raise ValueError("Token CSV contains an empty value.")
        tokens.append(int(value))
    if not tokens:
        raise ValueError("Token CSV must contain at least one token.")
    return tokens


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def ensure_dependencies() -> None:
    missing: list[str] = []
    for package in ("numpy", "torch", "transformers", "huggingface_hub"):
        if importlib.util.find_spec(package) is None:
            missing.append(package)

    if missing:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Install them before running."
        )


def download_model(model_id: str, model_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
        ],
    )
    return model_dir


def configure_hf_environment(work_dir: Path) -> None:
    hf_home = work_dir / ".hf_home"
    hf_cache = hf_home / "hub"
    hf_xet_cache = hf_home / "xet"
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    hf_xet_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_XET_CACHE"] = str(hf_xet_cache)
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def build_local_dumper(repo_root: Path) -> Path:
    source_dir = repo_root / "demo" / "transfomer-demo"
    build_dir = source_dir / "build-activation-diff"
    configure_cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DTRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP=OFF",
        "-DTRANSFOMER_DEMO_BUILD_TESTS=OFF",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    ]
    homebrew_openblas = Path("/opt/homebrew/opt/openblas")
    if homebrew_openblas.exists():
        configure_cmd.append(f"-DCMAKE_PREFIX_PATH={homebrew_openblas}")

    run(configure_cmd)
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "transfomer_activation_dump",
            "-j",
        ]
    )
    return build_dir / "transfomer_activation_dump"


def write_hf_manifest(
    output_dir: Path, tokens: list[int], activations: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"tokens": tokens, "activations": {}}
    for name, array in activations.items():
        file_name = f"{sanitize_name(name)}.npy"
        file_path = output_dir / file_name
        import numpy as np

        np.save(file_path, array.astype(np.float32, copy=False))
        manifest["activations"][name] = {
            "dtype": "float32",
            "shape": list(array.shape),
            "file": file_name,
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def capture_hf_activations(model_dir: Path, tokens: list[int]) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.eval()

    activations: dict[str, np.ndarray] = {}
    hooks = []

    def record(name: str):
        def _hook(_module, _inputs, output):
            tensor = output
            if isinstance(output, (tuple, list)):
                if not output:
                    raise RuntimeError(f"Hook output is empty for module {name}")
                tensor = output[0]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"Hook output for module {name} is not a Tensor: {type(tensor)}"
                )
            activations[name] = tensor.detach().cpu().to(torch.float32).numpy()

        return _hook

    hooks.append(model.model.embed_tokens.register_forward_hook(record("model.embed_tokens")))
    hooks.append(model.model.norm.register_forward_hook(record("model.norm")))

    for layer_index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_index}"
        hooks.append(layer.input_layernorm.register_forward_hook(record(f"{prefix}.input_layernorm")))
        hooks.append(layer.self_attn.register_forward_hook(record(f"{prefix}.self_attn")))
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                record(f"{prefix}.post_attention_layernorm")
            )
        )
        hooks.append(layer.mlp.register_forward_hook(record(f"{prefix}.mlp")))
        hooks.append(layer.register_forward_hook(record(f"{prefix}.output")))

    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)

    for hook in hooks:
        hook.remove()

    num_layers = len(model.model.layers)
    for layer_index in range(num_layers):
        prefix = f"model.layers.{layer_index}"
        hidden_in_name = "model.embed_tokens" if layer_index == 0 else f"model.layers.{layer_index - 1}.output"
        activations[f"{prefix}.post_attn_residual"] = (
            activations[hidden_in_name] + activations[f"{prefix}.self_attn"]
        ).astype(np.float32, copy=False)

    activations["last_hidden_state"] = activations["model.norm"]
    activations["logits"] = outputs.logits.detach().cpu().to(torch.float32).numpy()
    return activations


def load_local_activations(output_dir: Path) -> dict[str, Any]:
    import numpy as np

    manifest = json.loads((output_dir / "manifest.json").read_text())
    activations: dict[str, np.ndarray] = {}
    for name, meta in manifest["activations"].items():
        shape = tuple(int(dim) for dim in meta["shape"])
        file_path = output_dir / meta["file"]
        values = np.fromfile(file_path, dtype=np.float32)
        expected_numel = math.prod(shape)
        if values.size != expected_numel:
            raise RuntimeError(
                f"Local activation numel mismatch for {name}: got {values.size}, expected {expected_numel}"
            )
        activations[name] = values.reshape(shape)
    return activations


def compare_activations(
    hf_activations: dict[str, Any],
    local_activations: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    import numpy as np

    report: dict[str, Any] = {
        "thresholds": {"rtol": rtol, "atol": atol},
        "missing_in_local": [],
        "missing_in_hf": [],
        "compared_modules": [],
        "mismatched_modules": [],
    }

    hf_names = set(hf_activations.keys())
    local_names = set(local_activations.keys())
    report["missing_in_local"] = sorted(hf_names - local_names)
    report["missing_in_hf"] = sorted(local_names - hf_names)

    for name in sorted(hf_names & local_names):
        hf = hf_activations[name]
        local = local_activations[name]
        if tuple(hf.shape) != tuple(local.shape):
            report["mismatched_modules"].append(
                {
                    "name": name,
                    "reason": "shape_mismatch",
                    "hf_shape": list(hf.shape),
                    "local_shape": list(local.shape),
                }
            )
            continue

        diff = np.abs(hf - local)
        max_abs = float(diff.max()) if diff.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0
        allclose = bool(np.allclose(hf, local, rtol=rtol, atol=atol))

        item = {
            "name": name,
            "shape": list(hf.shape),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "allclose": allclose,
        }
        report["compared_modules"].append(item)
        if not allclose:
            report["mismatched_modules"].append(item)

    return report


def print_summary(report: dict[str, Any]) -> None:
    compared = len(report["compared_modules"])
    mismatched = len(report["mismatched_modules"])
    print(f"Compared modules: {compared}")
    print(f"Mismatched modules: {mismatched}")

    if report["missing_in_local"]:
        print("Missing in local:")
        for name in report["missing_in_local"]:
            print(f"  - {name}")
    if report["missing_in_hf"]:
        print("Missing in HF:")
        for name in report["missing_in_hf"]:
            print(f"  - {name}")

    if mismatched:
        print("Incorrect modules relative to HF:")
        for item in report["mismatched_modules"]:
            if item.get("reason") == "shape_mismatch":
                print(
                    f"  - {item['name']}: shape mismatch "
                    f"(hf={item['hf_shape']}, local={item['local_shape']})"
                )
            else:
                print(
                    f"  - {item['name']}: max_abs_diff={item['max_abs_diff']:.6g}, "
                    f"mean_abs_diff={item['mean_abs_diff']:.6g}"
                )
    else:
        print("All compared modules are within tolerance.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SmolLM2-135M-Instruct activations between HF and local deeptiny model."
    )
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HF model id to download.",
    )
    parser.add_argument(
        "--tokens",
        default="1,123,456,789,42,2",
        help="Comma-separated token ids for the forward pass.",
    )
    parser.add_argument(
        "--work-dir",
        default="tmp/smollm135m_activation_diff",
        help="Working directory for model files, activations, and report.",
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    ensure_dependencies()
    tokens = parse_tokens_csv(args.tokens)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    configure_hf_environment(work_dir)
    model_dir = work_dir / "model"
    hf_out_dir = work_dir / "hf_activations"
    local_out_dir = work_dir / "local_activations"
    report_path = work_dir / "report.json"

    repo_root = Path(__file__).resolve().parents[2]

    model_dir = download_model(args.model_id, model_dir)
    dumper_path = build_local_dumper(repo_root)

    hf_activations = capture_hf_activations(model_dir, tokens)
    write_hf_manifest(hf_out_dir, tokens, hf_activations)

    run(
        [
            str(dumper_path),
            str(model_dir),
            str(local_out_dir),
            ",".join(str(token) for token in tokens),
        ]
    )
    local_activations = load_local_activations(local_out_dir)

    report = compare_activations(
        hf_activations=hf_activations,
        local_activations=local_activations,
        rtol=args.rtol,
        atol=args.atol,
    )
    report["tokens"] = tokens
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report)
    print(f"Report written to: {report_path}")

    return 1 if report["mismatched_modules"] else 0


if __name__ == "__main__":
    sys.exit(main())
