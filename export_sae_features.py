"""
export_sae_features.py - Export SAE Feature Activations for Text Visualization

Exports per-token SAE feature activations to JSON for color-coded text highlighting in dashboard.
Shows which parts of text activate specific SAE features most strongly.

Dependencies: sae.py or sae_v3.py (SAE model), analyze_text.py (data loading)
Usage: uv run export_sae_features.py --ckpt sae_checkpoints_expert_router/sae_final.pt --feature 42 --samples 5000 --out feature_42.json
"""
import argparse
import json
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
def export_feature_activations(
    ckpt_path: str,
    feature_id: int,
    samples: int,
    k_keep: int,
    out_path: str,
    device: str = "cpu",
    tokenizer_path: str = "tokenizer_gpt_oss_20b"
):
    """
    Export activation scores for a specific SAE feature across text samples.

    Returns JSON with:
    - meta: feature_id, samples, d_sae
    - activations: list of {prompt, generated, tokens: [{text, score}], layer}
    """
    # Load SAE model
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Check if this is a v3 model with gate_mode
    gate_mode = ckpt.get("gate_mode", "topk")

    # Initialize model with appropriate parameters
    try:
        model = SAE(d_in=ckpt["d_in"], d_sae=ckpt["d_sae"], gate_mode=gate_mode).to(device)
    except TypeError:
        # Fallback for old SAE without gate_mode parameter
        model = SAE(d_in=ckpt["d_in"], d_sae=ckpt["d_sae"]).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    d_sae = ckpt["d_sae"]
    if feature_id >= d_sae:
        raise ValueError(f"feature_id {feature_id} >= d_sae {d_sae}")

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Group by (prompt, generated) to collect all tokens for each sample
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

        # Get activation score for this feature
        score = float(codes[0, feature_id].item())

        # Group by (prompt, gen) pair
        key = (prompt, gen)
        samples_dict[key]["token_scores"].append({"token_idx": t_idx, "score": score})
        samples_dict[key]["layer"] = layer
        samples_dict[key]["generated_text"] = gen

        seen += 1
        if seen >= samples:
            break

    # Convert to list and filter samples with at least one non-zero activation
    activations = []
    for (prompt, gen), data in samples_dict.items():
        # Check if any token has non-zero activation
        max_score = max(ts["score"] for ts in data["token_scores"])
        if max_score <= 0:
            continue

        # Tokenize the generated text to get actual token strings
        gen_text = data["generated_text"]
        if not gen_text:
            continue

        # Tokenize and map scores to actual tokens
        token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        # Create score map from token_idx to score
        score_map = {ts["token_idx"]: ts["score"] for ts in data["token_scores"]}

        # Get the actual range of token indices we have from the dataset
        token_indices = sorted(score_map.keys())
        if not token_indices:
            continue

        min_idx = min(token_indices)
        max_idx = max(token_indices)

        # DEBUG: Print for first example only
        if len(activations) == 0:
            print(f"\nDEBUG export_sae_features:")
            print(f"  len(token_strings) = {len(token_strings)}")
            print(f"  min_idx = {min_idx}, max_idx = {max_idx}")
            print(f"  Number of token_scores = {len(data['token_scores'])}")

        # The dataset token indices are offset from the start of generation
        # They directly correspond to positions in the tokenized text
        # So token_idx = min_idx corresponds to token_strings[0], etc.
        tokens_with_scores = []
        for i, tok_str in enumerate(token_strings):
            # The dataset token_idx for this position
            dataset_idx = min_idx + i
            # Get the score for this token_idx (if it exists in our data)
            score = score_map.get(dataset_idx, 0.0)
            tokens_with_scores.append({
                "text": tok_str,
                "score": score
            })

        # Don't truncate - send full text to match tokens
        activations.append({
            "prompt": prompt,
            "generated": gen_text,
            "tokens": tokens_with_scores,
            "max_score": max_score,
            "layer": data["layer"]
        })

    # Sort by max score descending
    activations.sort(key=lambda x: -x["max_score"])

    out = {
        "meta": {
            "feature_id": feature_id,
            "d_sae": d_sae,
            "samples": seen,
            "k_keep": k_keep,
            "checkpoint": ckpt_path
        },
        "activations": activations
    }

    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} (feature {feature_id}, {seen} samples scanned)")
    print(f"Samples with non-zero activations: {len(activations)}")
    if activations:
        print(f"Max activation: {activations[0]['max_score']:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="sae_checkpoints_expert_router/sae_final.pt")
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--k_keep", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    export_feature_activations(
        ckpt_path=args.ckpt,
        feature_id=args.feature,
        samples=args.samples,
        k_keep=args.k_keep,
        out_path=args.out,
        device=args.device
    )

if __name__ == "__main__":
    main()
