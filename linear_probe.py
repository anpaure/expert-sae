"""
linear_probe.py - PCA Baseline for Router Log-Probability Analysis

Fits PCA (with optional varimax rotation) on router log-probs as a linear baseline.
Provides component cards, top contexts, KL reconstruction metrics, and ablation analysis.

Dependencies: None (standalone)
Usage:
  uv run linear_probe.py fit --k 16 --samples 200000 --holdout 20000
  uv run linear_probe.py top-contexts --feature 3 --samples 120000 --keep 10
  uv run linear_probe.py cards --k 16 --samples 100000 --out feature_cards_lp.csv
  uv run linear_probe.py ablate --feature 3 --samples 16000
"""
import os, io, json, zipfile, pickle, argparse, csv, math
from typing import Dict, Any, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import list_repo_files, hf_hub_download

# --------------------
# Config (aligns with your sae.py defaults)
# --------------------
class CFG:
    hf_dataset_repo = "AmanPriyanshu/GPT-OSS-20B-MoE-expert-activations"
    layers_1_indexed = [21]          # choose [21] to match your current run
    all_layers = False               # set True to pull from all layers in files
    include_files = ["gpqa_diamond.zip"]   # keep it small/consistent for now

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

    save_dir = "./linear_probe_chkpts"
    os.makedirs(save_dir, exist_ok=True)

def _layer_keys() -> List[str]:
    return [f"layer_{i}" for i in range(64)] if CFG.all_layers \
        else [f"layer_{i - 1}" for i in CFG.layers_1_indexed]

# --------------------
# Dataset streaming (with text)
# --------------------
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

def _vectorize_logits(entry: Dict[str, Any], n_experts: int) -> Optional[torch.Tensor]:
    # Use log-probs (linear probe is on the same rep as your SAE)
    scores = entry.get("full_scores", None)
    if scores is None: return None
    p = torch.tensor(scores, dtype=torch.float32)
    return p.clamp_min(1e-12).log()  # log-probs

def iter_records_with_text(max_tokens: Optional[int]) -> Iterator[Tuple[torch.Tensor, str, str, int, int]]:
    """
    Yields: (logits_x, prompt, generated_only, token_idx, layer_1indexed)
    """
    files = list_repo_files(repo_id=CFG.hf_dataset_repo, repo_type="dataset")
    zips = [p for p in files if p.lower().endswith(".zip")]
    if CFG.include_files:
        zips = [p for p in zips if p in set(CFG.include_files)]

    layer_keys = _layer_keys()
    n_seen = 0
    n_experts_hint: Optional[int] = None

    def handle_obj(obj):
        nonlocal n_experts_hint, n_seen
        prompt = obj.get("prompt", "")
        gen = obj.get("generated_only", "")
        rd = obj.get("router_data", None)
        if not rd: return
        if not n_experts_hint:
            n_experts_hint = _infer_n_experts(obj, layer_keys)
            if not n_experts_hint: return
        for tb in rd:
            for tok_name, layers in tb.items():
                try:
                    t_idx = int(tok_name.split("_")[1])
                except Exception:
                    t_idx = -1
                for lk in layer_keys:
                    if lk not in layers: continue
                    x = _vectorize_logits(layers[lk], n_experts_hint)
                    if x is None: continue
                    z = int(lk.split("_")[1]) if "_" in lk else 0
                    yield (x, prompt, gen, t_idx, z + 1)
                    n_seen += 1
                    if max_tokens and n_seen >= max_tokens:
                        return

    for rel in zips:
        local = hf_hub_download(repo_id=CFG.hf_dataset_repo, repo_type="dataset", filename=rel)
        with zipfile.ZipFile(local, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"): continue
                if name.lower().endswith((".pkl", ".pickle")):
                    with zf.open(name, "r") as fh:
                        bio = io.BytesIO(fh.read())
                        try:
                            obj = pickle.load(bio); yield from handle_obj(obj)
                        except Exception:
                            bio.seek(0)
                            while True:
                                try:
                                    obj = pickle.load(bio)
                                except EOFError:
                                    break
                                except Exception:
                                    break
                                yield from handle_obj(obj)
                elif name.lower().endswith(".json"):
                    with zf.open(name, "r") as fh:
                        try:
                            obj = json.load(fh); yield from handle_obj(obj)
                        except Exception:
                            continue

# --------------------
# PCA / Varimax
# --------------------
@torch.no_grad()
def compute_mean_cov(samples: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Stream and compute mean and covariance of log-probs.
    Returns (mean[d], cov[d,d], n)
    """
    sum_x = None
    sum_xx = None
    n = 0
    for x, *_ in iter_records_with_text(max_tokens=samples):
        x = x.to(torch.float64)
        # re-normalize log-probs just in case
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        if sum_x is None:
            d = x.numel()
            sum_x = torch.zeros(d, dtype=torch.float64)
            sum_xx = torch.zeros(d, d, dtype=torch.float64)
        sum_x += x
        sum_xx += torch.outer(x, x)
        n += 1
    if n == 0:
        raise RuntimeError("No data found; check include_files / layers settings.")
    mean = sum_x / n
    cov = (sum_xx / n) - torch.outer(mean, mean)
    return mean.float(), cov.float(), n

@torch.no_grad()
def pca_from_cov(cov: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cov: [d,d]; returns (U_k[d,k], eigvals_k[k]) sorted desc.
    """
    # For symmetric PSD use eigh
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    k = min(k, evecs.shape[1])
    return evecs[:, :k].contiguous(), evals[:k].contiguous()

def varimax(P: torch.Tensor, q: int = 20, tol: float = 1e-6) -> torch.Tensor:
    """
    Orthogonal varimax rotation (classic). P: [d,k], returns rotated loadings [d,k].
    """
    d, k = P.shape
    R = torch.eye(k, device=P.device)
    last = 0.0
    for _ in range(q):
        B = P.t() @ (P * P * P - (P * P).mean(dim=0, keepdim=True) * P)
        U, S, Vt = torch.linalg.svd(B)
        R = U @ Vt
        P_rot = P @ R
        var = (P_rot**2).sum().item()
        if abs(var - last) < tol: break
        last = var
    return P @ R

# --------------------
# Projection / Recon / Metrics
# --------------------
@torch.no_grad()
def project(x: torch.Tensor, mean: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # x, mean: [d], U: [d,k]
    return (x - mean) @ U

@torch.no_grad()
def reconstruct(z: torch.Tensor, mean: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # z: [k], mean: [d], U: [d,k]
    return mean + (U @ z)

@torch.no_grad()
def eval_kl(mean: torch.Tensor, U: torch.Tensor, holdout: int = 20000) -> float:
    """
    KL(p || q) where p = softmax(renorm(x)), q = softmax(recon(x))
    """
    total = 0.0
    n = 0
    for x, *_ in iter_records_with_text(max_tokens=holdout):
        x = x.to(mean.device)
        log_p = x - torch.logsumexp(x, dim=-1, keepdim=True)
        z = project(x, mean, U)
        xh = reconstruct(z, mean, U)
        log_q = F.log_softmax(xh, dim=-1)
        # use log_target=True to stay precise
        total += float(F.kl_div(log_q, log_p, reduction="sum", log_target=True).item())
        n += 1
    return total / max(1, n)

# --------------------
# Cards, contexts, ablations
# --------------------
def write_csv(rows: List[dict], path: str):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)

@torch.no_grad()
def component_cards(mean: torch.Tensor, U: torch.Tensor, samples: int, keep: int = 20) -> List[dict]:
    """
    For each component j: activation stats and top +/- expert loadings; correlations with entropy / top1.
    """
    d, k = U.shape
    # Stats via streaming
    # We'll compute codes z_j, entropy H, top1 prob T1
    sum_z = torch.zeros(k)
    sum_z2 = torch.zeros(k)
    sum_zH = torch.zeros(k)
    sum_zT = torch.zeros(k)
    H_vals = []
    T_vals = []
    n = 0

    for x, *_ in iter_records_with_text(max_tokens=samples):
        x = x
        log_p = x - torch.logsumexp(x, dim=-1, keepdim=True)
        p = log_p.exp()
        H = float((-(p * log_p).sum()).item())
        T1 = float(p.max().item())
        z = project(x, mean, U)
        sum_z += z
        sum_z2 += z * z
        sum_zH += z * H
        sum_zT += z * T1
        H_vals.append(H); T_vals.append(T1)
        n += 1

    mean_z = sum_z / max(1, n)
    Ez2 = sum_z2 / max(1, n)
    var_z = (Ez2 - mean_z**2).clamp_min(1e-12)
    EzH = sum_zH / max(1, n)
    EzT = sum_zT / max(1, n)
    mean_H = torch.tensor(H_vals).mean().clamp_min(1e-12)
    mean_T = torch.tensor(T_vals).mean().clamp_min(1e-12)
    std_H = torch.tensor(H_vals).std().clamp_min(1e-6)
    std_T = torch.tensor(T_vals).std().clamp_min(1e-6)

    cov_zH = EzH - mean_z * mean_H
    cov_zT = EzT - mean_z * mean_T
    corr_H = (cov_zH / (var_z.sqrt() * std_H)).clamp(-1, 1)
    corr_T = (cov_zT / (var_z.sqrt() * std_T)).clamp(-1, 1)

    rows = []
    for j in range(k):
        col = U[:, j]
        # top + / - experts
        idx_pos = torch.topk(col, k=min(5, d)).indices.tolist()
        idx_neg = torch.topk(-col, k=min(5, d)).indices.tolist()
        rows.append({
            "component": j,
            "var_loading_norm": float(col.norm().item()),
            "corr_entropy": float(corr_H[j].item()),
            "corr_top1prob": float(corr_T[j].item()),
            "top_pos_experts": [(i, float(col[i].item())) for i in idx_pos],
            "top_neg_experts": [(i, float(col[i].item())) for i in idx_neg],
        })
    return rows

@torch.no_grad()
def top_contexts_for_component(j: int, mean: torch.Tensor, U: torch.Tensor,
                               samples: int, keep: int = 12, use_abs: bool = False):
    """
    Return top contexts by z_j (or |z_j| if use_abs).
    """
    heap: List[Tuple[float, Tuple[int,int,str,str]]] = []  # (score, (layer, token, prompt, gen))
    import heapq
    for x, pr, gen, t_idx, layer in iter_records_with_text(max_tokens=samples):
        z = project(x, mean, U)
        score = float(abs(z[j].item()) if use_abs else z[j].item())
        item = (score, (layer, t_idx, pr, gen))
        if len(heap) < keep:
            heapq.heappush(heap, item)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, item)
    return sorted(heap, key=lambda z: -z[0])

@torch.no_grad()
def ablate_component(j: int, mean: torch.Tensor, U: torch.Tensor, samples: int = 8192) -> List[Tuple[int, float]]:
    """
    Set z_j=0 and report mean Δprob over experts.
    """
    d, k = U.shape
    delta_sum = torch.zeros(d)
    n = 0
    for x, *_ in iter_records_with_text(max_tokens=samples):
        log_p = x - torch.logsumexp(x, dim=-1, keepdim=True)
        p = log_p.exp()

        z = project(x, mean, U)
        xh = reconstruct(z, mean, U)
        log_q = F.log_softmax(xh, dim=-1)
        q = log_q.exp()

        z_abl = z.clone()
        z_abl[j] = 0.0
        xh2 = reconstruct(z_abl, mean, U)
        q2 = F.log_softmax(xh2, dim=-1).exp()

        delta_sum += (q2 - q)
        n += 1
    delta = delta_sum / max(1, n)
    top = torch.topk(delta.abs(), k=8)
    return [(int(i), float(delta[i].item())) for i in top.indices]

# --------------------
# CLI
# --------------------
def save_probe(mean: torch.Tensor, U: torch.Tensor, eval_kl_value: float, tag: str):
    obj = {
        "mean": mean.cpu(),
        "U": U.cpu(),
        "eval_kld": eval_kl_value,
    }
    path = os.path.join(CFG.save_dir, f"linear_probe_{tag}.pt")
    torch.save(obj, path)
    print(f"Saved: {path}")

def load_probe(path: str):
    obj = torch.load(path, map_location="cpu")
    return obj["mean"], obj["U"], obj.get("eval_kld", None)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="fit PCA (+ optional varimax), report KL, save probe")
    p_fit.add_argument("--k", type=int, default=16)
    p_fit.add_argument("--samples", type=int, default=200000)
    p_fit.add_argument("--holdout", type=int, default=20000)
    p_fit.add_argument("--varimax", action="store_true", help="apply varimax rotation")
    p_fit.add_argument("--tag", type=str, default="k16")

    p_cards = sub.add_parser("cards", help="write component cards CSV")
    p_cards.add_argument("--probe", type=str, default=None, help="path to saved probe; if omitted, fits quickly on the fly")
    p_cards.add_argument("--k", type=int, default=16)
    p_cards.add_argument("--samples", type=int, default=100000)
    p_cards.add_argument("--varimax", action="store_true")
    p_cards.add_argument("--out", type=str, default=os.path.join(CFG.save_dir, "feature_cards_lp.csv"))

    p_ctx = sub.add_parser("top-contexts", help="print top contexts for a component")
    p_ctx.add_argument("--probe", type=str, default=None)
    p_ctx.add_argument("--k", type=int, default=16)
    p_ctx.add_argument("--feature", type=int, required=True)
    p_ctx.add_argument("--samples", type=int, default=120000)
    p_ctx.add_argument("--keep", type=int, default=8)
    p_ctx.add_argument("--varimax", action="store_true")
    p_ctx.add_argument("--abs", action="store_true", help="rank by |z_j|")

    p_abl = sub.add_parser("ablate", help="ablate one component, report Δprob")
    p_abl.add_argument("--probe", type=str, default=None)
    p_abl.add_argument("--k", type=int, default=16)
    p_abl.add_argument("--feature", type=int, required=True)
    p_abl.add_argument("--samples", type=int, default=8192)
    p_abl.add_argument("--varimax", action="store_true")

    args = parser.parse_args()

    device = CFG.device
    print(f"Device: {device}")

    if args.cmd == "fit":
        mean, cov, n = compute_mean_cov(args.samples)
        U, evals = pca_from_cov(cov, args.k)
        if args.varimax:
            U = varimax(U)
        kld = eval_kl(mean, U, holdout=args.holdout)
        exp_var = evals.sum() / cov.trace().clamp_min(1e-12)
        print(f"k={U.shape[1]} | heldout KL ≈ {kld:.5f} | explained variance ≈ {float(exp_var):.3f}")
        save_probe(mean, U, kld, args.tag + ("_vx" if args.varimax else ""))

    elif args.cmd == "cards":
        if args.probe:
            mean, U, _ = load_probe(args.probe)
        else:
            mean, cov, n = compute_mean_cov(args.samples)
            U, _ = pca_from_cov(cov, args.k)
            if args.varimax: U = varimax(U)
        rows = component_cards(mean, U, samples=args.samples, keep=20)
        write_csv(rows, args.out)
        print(f"Wrote {args.out} with {len(rows)} rows")

    elif args.cmd == "top-contexts":
        if args.probe:
            mean, U, _ = load_probe(args.probe)
        else:
            mean, cov, n = compute_mean_cov(args.samples)
            U, _ = pca_from_cov(cov, args.k)
            if args.varimax: U = varimax(U)
        top = top_contexts_for_component(args.feature, mean, U, samples=args.samples, keep=args.keep, use_abs=args.abs)
        print(f"\n=== Top {len(top)} contexts for component {args.feature} ===")
        for score, (layer, tok, pr, gen) in top:
            print(f"\n[Layer {layer} | token {tok} | z={score:.3f}]")
            print("PROMPT:", (pr[:400] + ("..." if len(pr) > 400 else "")))
            print("GENERATED:", (gen[:800] + ("..." if len(gen) > 800 else "")))

    elif args.cmd == "ablate":
        if args.probe:
            mean, U, _ = load_probe(args.probe)
        else:
            mean, cov, n = compute_mean_cov(args.samples)
            U, _ = pca_from_cov(cov, args.k)
            if args.varimax: U = varimax(U)
        topchg = ablate_component(args.feature, mean, U, samples=args.samples)
        print(f"Top |Δp| experts when ablating comp {args.feature}:", topchg)

if __name__ == "__main__":
    main()
# # 1) Fit a 16-dim linear probe and report KL on a 20k holdout
# uv run linear_probe.py fit --k 16 --samples 200000 --holdout 20000

# # (Optional) try varimax for sparser expert loadings
# uv run linear_probe.py fit --k 16 --samples 200000 --holdout 20000 --varimax --tag k16_vx

# # 2) Dump component “cards” (expert loadings + correlations) to CSV
# uv run linear_probe.py cards --k 16 --samples 120000 --out ./linear_cards.csv

# # 3) Read the top text contexts for a specific component
# uv run linear_probe.py top-contexts --feature 3 --samples 120000 --keep 10

# # 4) See which experts a component pushes probability to/from (ablate it)
# uv run linear_probe.py ablate --feature 3 --samples 16000