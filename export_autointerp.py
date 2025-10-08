"""
export_autointerp.py - Export SAE Features for Automated Interpretation

Exports top-activating examples per feature in a format optimized for automated interpretation.
Shows specific words/phrases where each feature activates most strongly.

Dependencies: sae_v3.py (SAE model), analyze_text.py (data loading), transformers (tokenizer)
Usage: uv run export_autointerp.py --ckpt sae_checkpoints_expert_router_v3/sae_final.pt --features 0,2,3,5,10 --out-dir autointerp_exports
"""
import argparse
import json
import os
import torch
from typing import List, Tuple, Dict, Any
from analyze_text import iter_records_with_text
from transformers import AutoTokenizer

# Try to import from sae_v3 first (has JumpReLU), fallback to sae.py
try:
    from sae_v3 import SAE
except ImportError:
    from sae import SAE


@torch.no_grad()
def export_feature_for_autointerp(
    ckpt_path: str,
    feature_id: int,
    samples: int,
    k_keep: int,
    top_k: int,
    device: str = "cpu",
    tokenizer_path: str = "tokenizer_gpt_oss_20b"
) -> Dict[str, Any]:
    """
    Export top activating examples for a single feature.

    Returns a dict with:
    - feature_id: int
    - top_activations: list of {text, activation_score, highlighted_tokens}
    """
    # Load SAE model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gate_mode = ckpt.get("gate_mode", "topk")

    try:
        model = SAE(d_in=ckpt["d_in"], d_sae=ckpt["d_sae"], gate_mode=gate_mode).to(device)
    except TypeError:
        model = SAE(d_in=ckpt["d_in"], d_sae=ckpt["d_sae"]).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    d_sae = ckpt["d_sae"]
    if feature_id >= d_sae:
        raise ValueError(f"feature_id {feature_id} >= d_sae {d_sae}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Collect all activations
    from collections import defaultdict
    samples_dict = defaultdict(lambda: {"token_scores": [], "layer": None, "generated_text": ""})
    seen = 0

    for vec, prompt, gen, t_idx, layer in iter_records_with_text(layers_override=None, max_tokens=samples):
        x = vec.to(device).unsqueeze(0)

        # Compute SAE codes with Top-K sparsification
        codes = torch.relu(model.enc(x))
        vals, _ = torch.topk(codes, k=min(k_keep, codes.shape[1]), dim=1)
        kth = vals[:, -1].unsqueeze(1)
        mask = (codes >= kth).to(codes.dtype)
        codes = codes * mask

        score = float(codes[0, feature_id].item())

        if score > 0:  # Only keep non-zero activations
            key = (prompt, gen)
            samples_dict[key]["token_scores"].append({"token_idx": t_idx, "score": score})
            samples_dict[key]["layer"] = layer
            samples_dict[key]["generated_text"] = gen

        seen += 1
        if seen >= samples:
            break

    # Process and rank activations
    activations = []
    for (prompt, gen), data in samples_dict.items():
        if not data["generated_text"]:
            continue

        max_score = max(ts["score"] for ts in data["token_scores"])

        # Tokenize the generated text
        token_ids = tokenizer.encode(data["generated_text"], add_special_tokens=False)
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        # Create score map
        score_map = {ts["token_idx"]: ts["score"] for ts in data["token_scores"]}
        token_indices = sorted(score_map.keys())
        if not token_indices:
            continue

        min_idx = min(token_indices)

        # Find tokens with highest activations
        highlighted = []
        for i, tok_str in enumerate(token_strings):
            actual_idx = min_idx + i
            score = score_map.get(actual_idx, 0.0)
            if score > 0:
                highlighted.append({
                    "token": tok_str,
                    "position": i,
                    "score": round(score, 4)
                })

        # Sort highlighted tokens by score
        highlighted.sort(key=lambda x: -x["score"])

        activations.append({
            "prompt": prompt[:500],
            "generated_text": data["generated_text"][:500],
            "max_activation": round(max_score, 4),
            "top_tokens": highlighted[:10],  # Top 10 tokens in this example
            "full_text": f"{prompt[:200]}... → {data['generated_text'][:300]}"
        })

    # Sort by max activation and take top-k
    activations.sort(key=lambda x: -x["max_activation"])
    top_activations = activations[:top_k]

    return {
        "feature_id": feature_id,
        "num_examples": len(top_activations),
        "max_activation_overall": top_activations[0]["max_activation"] if top_activations else 0,
        "examples": top_activations
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="sae_checkpoints_expert_router_v3/sae_final.pt")
    ap.add_argument("--features", type=str, required=True, help="Comma-separated feature IDs")
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--k_keep", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=20, help="Top K examples per feature")
    ap.add_argument("--out-dir", type=str, default="autointerp_exports")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    # Parse feature IDs
    try:
        feature_ids = [int(f.strip()) for f in args.features.split(",")]
    except ValueError:
        print("ERROR: --features must be comma-separated integers")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Exporting {len(feature_ids)} features for auto-interpretation")
    print(f"Features: {feature_ids}")
    print(f"Top {args.top_k} examples per feature")
    print()

    results = []

    for i, feature_id in enumerate(feature_ids, 1):
        print(f"[{i}/{len(feature_ids)}] Processing feature {feature_id}...")

        feature_data = export_feature_for_autointerp(
            ckpt_path=args.ckpt,
            feature_id=feature_id,
            samples=args.samples,
            k_keep=args.k_keep,
            top_k=args.top_k,
            device=args.device
        )

        # Save individual feature file
        out_file = os.path.join(args.out_dir, f"feature_{feature_id}_autointerp.json")
        with open(out_file, "w") as f:
            json.dump(feature_data, f, ensure_ascii=False, indent=2)

        print(f"  Saved {out_file}")
        print(f"  {feature_data['num_examples']} examples, max activation: {feature_data['max_activation_overall']:.4f}")

        results.append(feature_data)

    # Save combined file
    combined_file = os.path.join(args.out_dir, "all_features_autointerp.json")
    with open(combined_file, "w") as f:
        json.dump({
            "features": results,
            "metadata": {
                "checkpoint": args.ckpt,
                "samples_scanned": args.samples,
                "top_k_per_feature": args.top_k
            }
        }, f, ensure_ascii=False, indent=2)

    print()
    print(f"✓ Done! Saved {len(feature_ids)} feature files to {args.out_dir}/")
    print(f"  Combined file: {combined_file}")


if __name__ == "__main__":
    main()