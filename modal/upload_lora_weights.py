#!/usr/bin/env python3
"""
Upload LoRA weights to the Modal voxcpm-lora volume.

Usage:
  # From RunPod where checkpoints/latest/lora_weights.safetensors exists:
  python modal/upload_lora_weights.py --lora-dir /workspace/checkpoints/latest

  # Or from any machine with the weights:
  python modal/upload_lora_weights.py --lora-dir /path/to/lora_dir
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA weights to Modal volume")
    parser.add_argument(
        "--lora-dir",
        type=str,
        required=True,
        help="Directory containing lora_weights.safetensors (and optionally lora_config.json)",
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="voxcpm-lora",
        help="Modal volume name (default: voxcpm-lora)",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="/latest",
        help="Remote path in volume (default: /latest)",
    )
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    if not lora_dir.exists():
        print(f"Error: {lora_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    safetensors = lora_dir / "lora_weights.safetensors"
    if not safetensors.exists():
        print(f"Error: {safetensors} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading LoRA weights from {lora_dir} to volume {args.volume}:{args.remote_path}")

    # Upload lora_weights.safetensors
    cmd = [
        "modal", "volume", "put",
        args.volume,
        str(safetensors),
        f"{args.remote_path}/lora_weights.safetensors",
    ]
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Upload lora_config.json if it exists
    config_file = lora_dir / "lora_config.json"
    if config_file.exists():
        cmd = [
            "modal", "volume", "put",
            args.volume,
            str(config_file),
            f"{args.remote_path}/lora_config.json",
        ]
        print(f"  > {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print("  No lora_config.json found, will use default config (r=32, alpha=16, lm+dit)")

    print("\nDone! LoRA weights uploaded to Modal volume.")
    print(f"\nTo deploy the endpoint:")
    print(f"  modal deploy modal/mindexpander_voxcpm_modal_app.py")
    print(f"\nTo test:")
    print(f"  curl -X POST https://YOUR_URL/v1/audio/speech \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model\":\"mindexpander-voxcpm2\",\"input\":\"Hello from the clone\",\"voice\":\"default\"}}' \\")
    print(f"    --output test.wav")


if __name__ == "__main__":
    main()
