"""
export_all_features.py - Batch Export Multiple SAE Features

Exports multiple SAE features at once to a designated folder for multi-feature visualization.

Dependencies: export_sae_features.py
Usage: uv run export_all_features.py --features 5,10,15,42 --samples 50000 --out-dir feature_exports
"""
import argparse
import subprocess
from pathlib import Path

def export_all_features(
    ckpt_path: str,
    feature_ids: list[int],
    samples: int,
    k_keep: int,
    out_dir: str,
    device: str = "cpu"
):
    """Export multiple features to a directory."""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    print(f"Exporting {len(feature_ids)} features to {out_dir}/")
    print(f"Features: {feature_ids}")
    print(f"Samples per feature: {samples}")
    print()

    for i, feature_id in enumerate(feature_ids, 1):
        out_file = out_path / f"feature_{feature_id}.json"
        print(f"[{i}/{len(feature_ids)}] Exporting feature {feature_id}...")

        cmd = [
            "uv", "run", "export_sae_features.py",
            "--ckpt", ckpt_path,
            "--feature", str(feature_id),
            "--samples", str(samples),
            "--k_keep", str(k_keep),
            "--out", str(out_file),
            "--device", device
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
        else:
            print(f"  {result.stdout.strip()}")

    print()
    print(f"✓ Done! Exported {len(feature_ids)} features to {out_dir}/")
    print(f"  Load multiple features in dashboard for comparison")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="sae_checkpoints_expert_router/sae_final.pt")
    ap.add_argument("--features", type=str, required=True, help="Comma-separated feature IDs (e.g., 5,10,15,42)")
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--k_keep", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default="feature_exports")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    # Parse feature IDs
    try:
        feature_ids = [int(f.strip()) for f in args.features.split(",")]
    except ValueError:
        print("ERROR: --features must be comma-separated integers (e.g., 5,10,15,42)")
        return

    export_all_features(
        ckpt_path=args.ckpt,
        feature_ids=feature_ids,
        samples=args.samples,
        k_keep=args.k_keep,
        out_dir=args.out_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()
