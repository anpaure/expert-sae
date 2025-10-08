# analyze_text.py - Text-based SAE Feature Interpretation (patched)
#
# Usage examples:
#   uv run analyze_text.py --ckpt sae_checkpoints_expert_router/sae_final.pt --feature 10
#   uv run analyze_text.py --ckpt ... --feature 3 --samples 120000 --keep 10 --codes post
#   uv run analyze_text.py --ckpt ... --feature 7 --layers 21 24 --k 8
#
import os, io, json, zipfile, pickle, argparse, heapq
from typing import Dict, Any, Iterator, List, Optional, Tuple
import torch
import torch.nn.functional as F
from huggingface_hub import list_repo_files, hf_hub_download

from sae import cfg, SAE  # import your cfg + SAE class

# -------------------------
# Helpers
# -------------------------
def _safe_log_probs(scores: List[float]) -> torch.Tensor:
    p = torch.tensor(scores, dtype=torch.float32)
    return p.clamp_min(1e-12).log()

def _infer_n_experts(obj: Dict[str, Any], layer_keys: List[str]) -> int:
    rd = obj.get("router_data", None)
    if not rd: return 0
    for tb in rd:
        for _tok, layers in tb.items():
            for lk in layer_keys:
                if lk in layers:
                    e = layers[lk]
                    fs = e.get("full_scores", None)
                    if fs is not None: return len(fs)
                    sel = e.get("selected_experts", [])
                    if sel: return max(sel) + 1
    return 0

def _layer_keys(layers_override: Optional[List[int]]) -> List[str]:
    if layers_override:
        return [f"layer_{i-1}" for i in layers_override]
    return [f"layer_{i}" for i in range(64)] if cfg.all_layers else [f"layer_{i-1}" for i in cfg.layers_1_indexed]

# -------------------------
# Dataset iterator (with text)
# -------------------------
def iter_records_with_text(
    layers_override: Optional[List[int]],
    max_tokens: Optional[int] = 200_000
) -> Iterator[Tuple[torch.Tensor, str, str, int, int]]:
    """
    Yields: (vec_log_probs, prompt, generated_text, token_idx_in_gen, layer_1indexed)
    """
    files = list_repo_files(repo_id=cfg.hf_dataset_repo, repo_type="dataset")
    zips = [p for p in files if p.lower().endswith(".zip")]
    if cfg.include_files:
        zips = [p for p in zips if p in set(cfg.include_files)]

    n_seen = 0
    layer_keys = _layer_keys(layers_override)
    n_experts_hint: Optional[int] = None

    def handle_obj(obj):
        nonlocal n_experts_hint, n_seen
        prompt = obj.get("prompt", "")
        # Correct field in the dataset; provide fallbacks just in case
        gen = (obj.get("generated_text", None) or
               obj.get("generated_only", None) or
               obj.get("response", "")) or ""
        rd = obj.get("router_data", None)
        if not rd: return
        if not n_experts_hint:
            n_experts_hint = _infer_n_experts(obj, layer_keys)
            if not n_experts_hint: return

        for tb in rd:
            for tok_name, layers in tb.items():
                # token name like "token_183" → 183 (dataset sometimes starts at the first generated token)
                try:
                    t_idx = int(tok_name.split("_")[1])
                except Exception:
                    t_idx = -1
                for lk in layer_keys:
                    e = layers.get(lk)
                    if not e: 
                        continue
                    scores = e.get("full_scores", None)
                    if scores is None:
                        continue
                    vec = _safe_log_probs(scores)
                    layer_1idx = (int(lk.split("_")[1]) + 1) if "_" in lk else 1
                    yield (vec, prompt, gen, t_idx, layer_1idx)
                    n_seen += 1
                    if max_tokens is not None and n_seen >= max_tokens:
                        return

    for rel in zips:
        local = hf_hub_download(repo_id=cfg.hf_dataset_repo, repo_type="dataset", filename=rel)
        with zipfile.ZipFile(local, "r") as zf:
            for name in zf.namelist():
                if name.endswith('/'):
                    continue
                if name.lower().endswith((".pkl", ".pickle")):
                    with zf.open(name, "r") as fh:
                        bio = io.BytesIO(fh.read())
                        try:
                            obj = pickle.load(bio)
                            for rec in handle_obj(obj):
                                yield rec
                                if max_tokens and n_seen >= max_tokens: return
                        except Exception:
                            bio.seek(0)
                            while True:
                                try:
                                    obj = pickle.load(bio)
                                except EOFError:
                                    break
                                except Exception:
                                    break
                                for rec in handle_obj(obj):
                                    yield rec
                                    if max_tokens and n_seen >= max_tokens: return
                elif name.lower().endswith(".json"):
                    with zf.open(name, "r") as fh:
                        try:
                            obj = json.load(fh)
                            for rec in handle_obj(obj):
                                yield rec
                                if max_tokens and n_seen >= max_tokens: return
                        except Exception:
                            continue

# -------------------------
# Load model + probe k from ckpt
# -------------------------
@torch.no_grad()
def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SAE(d_in=ckpt["d_in"], d_sae=ckpt["d_sae"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    cfg_from_ckpt = ckpt.get("config", {})
    return model, cfg_from_ckpt

def _resolve_k_for_analysis(args_k: Optional[int], cfg_from_ckpt: Dict[str, Any]) -> Tuple[bool, int]:
    use_topk = bool(cfg_from_ckpt.get("use_topk", True))
    k_ckpt = int(cfg_from_ckpt.get("topk_end", cfg_from_ckpt.get("topk", 0)))
    if args_k is not None:
        return use_topk, int(args_k)
    return use_topk, int(k_ckpt or 0)

# -------------------------
# Top contexts
# -------------------------
@torch.no_grad()
def top_contexts_for_feature(
    j: int, model: SAE, device: str, use_topk: bool, k_keep: int,
    samples: int, keep: int, layers_override: Optional[List[int]], codes_mode: str
):
    """
    Returns the top-`keep` (score, layer, token_idx, prompt, generated_text) by code value of feature j.
    codes_mode: "pre" (ReLU only) or "post" (after TopK gating). Default "post".
    """
    heap: List[Tuple[float, Tuple[int,int,str,str]]] = []  # min-heap

    for vec, prompt, gen, t_idx, layer in iter_records_with_text(layers_override=layers_override, max_tokens=samples):
        x = vec.to(device).unsqueeze(0)

        if codes_mode == "pre":
            codes = F.relu(model.enc(x))
            score = float(codes[0, j].item())
        else:
            # post-TopK (match training behavior)
            recon, codes, mask = model(x, k=k_keep if use_topk and k_keep > 0 else None)
            score = float(codes[0, j].item())

        if score <= 0:
            continue
        item = (score, (layer, t_idx, prompt, gen))
        if len(heap) < keep:
            heapq.heappush(heap, item)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, item)

    return sorted(heap, key=lambda z: -z[0])

# -------------------------
# Streaming correlations
# -------------------------
@torch.no_grad()
def streaming_corrs(
    model: SAE, device: str, samples: int, use_topk: bool, k_keep: int,
    layers_override: Optional[List[int]], codes_mode: str
):
    """
    Pearson corr(feature_j, entropy) and corr(feature_j, top1prob), computed streaming (no big [N,d] buffer).
    Returns two tensors of shape [d_sae].
    """
    # We'll maintain means and second moments online (Welford-like).
    # For covariance: E[f*H] - E[f]*E[H], etc.
    first_vec = True
    d_sae = model.dec.weight.shape[1]
    Ef = torch.zeros(d_sae)
    Ef2 = torch.zeros(d_sae)
    EfH = torch.zeros(d_sae)
    EfT = torch.zeros(d_sae)
    EH = 0.0; EH2 = 0.0; ET = 0.0; ET2 = 0.0
    n = 0

    it = iter_records_with_text(layers_override=layers_override, max_tokens=samples)
    for vec, *_ in it:
        x = vec.to(device).unsqueeze(0)
        # entropy + top1 prob from normalized log-probs
        log_p = x - torch.logsumexp(x, dim=-1, keepdim=True)
        p = log_p.exp()
        H = float((-(p * log_p).sum(dim=-1)).item())
        T = float(p.max(dim=-1).values.item())

        if codes_mode == "pre":
            codes = F.relu(model.enc(x)).squeeze(0).cpu()
        else:
            _recon, codes, _mask = model(x, k=k_keep if use_topk and k_keep > 0 else None)
            codes = codes.squeeze(0).cpu()

        # online stats
        Ef += codes
        Ef2 += codes * codes
        EfH += codes * H
        EfT += codes * T
        EH += H; EH2 += H*H
        ET += T; ET2 += T*T
        n += 1

    if n == 0:
        d = model.dec.weight.shape[1]
        return torch.zeros(d), torch.zeros(d)

    Ef /= n; Ef2 /= n; EfH /= n; EfT /= n
    EH /= n; ET /= n; EH2 /= n; ET2 /= n

    var_f = (Ef2 - Ef * Ef).clamp_min(1e-12)
    var_H = max(EH2 - EH*EH, 1e-12)
    var_T = max(ET2 - ET*ET, 1e-12)

    cov_fH = EfH - Ef * EH
    cov_fT = EfT - Ef * ET

    corr_H = cov_fH / (var_f.sqrt() * (var_H ** 0.5))
    corr_T = cov_fT / (var_f.sqrt() * (var_T ** 0.5))
    return corr_H.clamp(-1,1), corr_T.clamp(-1,1)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(cfg.save_dir, "sae_final.pt"))
    ap.add_argument("--feature", type=int, required=True, help="feature index j to inspect")
    ap.add_argument("--samples", type=int, default=120000)
    ap.add_argument("--keep", type=int, default=12, help="how many contexts to show")
    ap.add_argument("--k", type=int, default=None, help="override TopK used at train time (default: from ckpt)")
    ap.add_argument("--codes", choices=["pre","post"], default="post", help="use pre-TopK or post-TopK codes")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="1-indexed layers to include (override cfg)")
    args = ap.parse_args()

    device = cfg.device
    model, cfg_from_ckpt = load_model(args.ckpt, device)
    use_topk, k_keep = _resolve_k_for_analysis(args.k, cfg_from_ckpt)

    # Top contexts
    top = top_contexts_for_feature(
        args.feature, model, device, use_topk, k_keep,
        samples=args.samples, keep=args.keep, layers_override=args.layers, codes_mode=args.codes
    )
    print(f"\n=== Top {len(top)} contexts for feature {args.feature} ({args.codes}-TopK) ===")
    for score, (layer, t_idx, prompt, gen) in top:
        print(f"\n[Layer {layer} | token {t_idx} | code={score:.3f}]")
        print("PROMPT:", (prompt[:400] + ("..." if len(prompt) > 400 else "")))
        print("GENERATED:", (gen[:800] + ("..." if len(gen) > 800 else "")))

    # Correlations (streaming; matches train-time normalization for H/T)
    corr_H, corr_T = streaming_corrs(
        model, device, samples=min(50000, args.samples),
        use_topk=use_topk, k_keep=k_keep,
        layers_override=args.layers, codes_mode=args.codes
    )
    preview = min(10, corr_H.numel())
    print("\nCorrelation preview (first 10 features):")
    print("corr_entropy[:10]:", [round(v, 3) for v in corr_H[:preview].tolist()])
    print("corr_top1prob[:10]:", [round(v, 3) for v in corr_T[:preview].tolist()])

if __name__ == "__main__":
    main()
