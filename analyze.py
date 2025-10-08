"""
analyze.py - SAE Feature Analysis & Correlation Metrics

Analyzes trained SAE models: computes reconstruction metrics, feature correlations with router
properties (entropy, top-p), and identifies top-activating experts/tokens per feature.

Dependencies: sae.py (imports ExpertRouterIterable, SAE, cfg, iter_all_vectors)
Usage: uv run analyze.py --ckpt sae_checkpoints_*/sae_final.pt --samples 100000
"""
import os, io, json, zipfile, pickle, math, random, argparse, csv
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F

# ---- import your modules from sae.py ----
# If sae.py is in the same folder, we can import the dataset + model classes.
# Otherwise, copy the needed bits or adjust your PYTHONPATH.
from sae import ExpertRouterIterable, SAE, cfg, iter_all_vectors

@torch.no_grad()
def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in = ckpt["d_in"]; d_sae = ckpt["d_sae"]
    model = SAE(d_in=d_in, d_sae=d_sae).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, d_in, d_sae

def batch_iter(n_examples: int, batch_size: int = 4096):
    """Recreate the same stream you trained on (logits mode)."""
    # We do NOT normalize for logits here (consistent with training).
    dataset = ExpertRouterIterable(
        max_tokens=n_examples,
        normalize_inputs=False,
        normalize_per_layer=cfg.normalize_per_layer
    )
    buf = []
    for _l1, vec in iter_all_vectors(max_tokens=n_examples):
        buf.append(vec)
        if len(buf) == batch_size:
            yield torch.stack(buf, dim=0)
            buf = []
    if buf:
        yield torch.stack(buf, dim=0)

def cosine(a, b, dim=-1, eps=1e-8):
    a = F.normalize(a, dim=dim)
    b = F.normalize(b, dim=dim)
    return (a * b).sum(dim=dim)

@torch.no_grad()
def compute_feature_stats(model: SAE, device: str, n_examples: int = 100_000, topk: Optional[int] = None):
    """
    Returns:
      stats: dict with activation_rate, mean_when_active, corr_with_entropy, corr_with_top1prob
      W: decoder weight matrix [d_in, d_sae]
    """
    d_in = model.dec.weight.shape[0]
    d_sae = model.dec.weight.shape[1]
    total = 0
    hits = torch.zeros(d_sae)
    mag_sum = torch.zeros(d_sae)
    # correlations
    cov_feat_entropy = torch.zeros(d_sae)
    cov_feat_top1 = torch.zeros(d_sae)
    mean_entropy = 0.0
    mean_top1 = 0.0

    # Means for feature activation (for covariance)
    mean_feat = torch.zeros(d_sae)

    for xb in batch_iter(n_examples):
        xb = xb.to(device)
        # logits mode: xb are log-probs; turn into probs for entropy
        log_p = xb
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1)  # H(p)
        top1 = p.max(dim=-1).values

        # pre-TopK codes (natural)
        pre = torch.relu(model.enc(xb))
        # post-TopK (like training)
        if topk is not None and topk > 0:
            k = min(topk, pre.shape[1])
            vals, _ = torch.topk(pre, k=k, dim=1)
            kth = vals[:, -1].unsqueeze(1)
            mask = (pre >= kth).to(pre.dtype)
            codes = pre * mask
        else:
            codes = pre

        # stats
        active = (codes > 0)
        hits += active.sum(dim=0).cpu()
        mag_sum += codes.sum(dim=0).cpu()
        mean_feat += codes.mean(dim=0).cpu()
        mean_entropy += entropy.mean().item() * xb.shape[0]
        mean_top1 += top1.mean().item() * xb.shape[0]

        # covariances feature vs entropy/top1
        # (E[feat*metric] - E[feat]E[metric]) computed later; accumulate E[feat*metric]
        cov_feat_entropy += (codes * entropy.unsqueeze(1)).mean(dim=0).cpu() * xb.shape[0]
        cov_feat_top1    += (codes * top1.unsqueeze(1)).mean(dim=0).cpu() * xb.shape[0]

        total += xb.shape[0]

    act_rate = hits / total
    mean_when_active = mag_sum / (hits.clamp_min(1))
    mean_feat = mean_feat / 1.0  # already mean over batches
    mean_entropy /= total
    mean_top1 /= total
    # finish covariance (convert E[xy] to Cov if you want Pearson later)
    E_feat_entropy = cov_feat_entropy / total
    E_feat_top1    = cov_feat_top1 / total
    cov_f_H = E_feat_entropy - mean_feat * mean_entropy
    cov_f_T = E_feat_top1 - mean_feat * mean_top1

    # simple scale for correlation: std(feature) ~ from codes; here approximate var by E[feat^2] - E[feat]^2
    # compute E[feat^2]
    E_feat2 = torch.zeros(d_sae)
    counted = 0
    for xb in batch_iter(min(n_examples, 50_000)):  # smaller pass
        xb = xb.to(device)
        pre = torch.relu(model.enc(xb))
        if topk is not None and topk > 0:
            k = min(topk, pre.shape[1])
            vals, _ = torch.topk(pre, k=k, dim=1)
            kth = vals[:, -1].unsqueeze(1)
            mask = (pre >= kth).to(pre.dtype)
            codes = pre * mask
        else:
            codes = pre
        E_feat2 += (codes**2).mean(dim=0).cpu()
        counted += 1
    E_feat2 /= max(1, counted)
    var_feat = (E_feat2 - mean_feat**2).clamp_min(1e-12)

    # approximate std(entropy), std(top1) from the first pass
    # re-compute quickly on a subset
    H_vals = []
    T_vals = []
    for xb in batch_iter(min(n_examples, 20_000)):
        xb = xb.to(device)
        log_p = xb - torch.logsumexp(xb, dim=-1, keepdim=True)
        p = log_p.exp()
        H_vals.append((-(p * log_p).sum(dim=-1)).cpu())
        T_vals.append(p.max(dim=-1).values.cpu())
    H = torch.cat(H_vals); T = torch.cat(T_vals)
    std_H = H.std().clamp_min(1e-6)
    std_T = T.std().clamp_min(1e-6)

    corr_feat_entropy = (cov_f_H / (var_feat.sqrt() * std_H)).clamp(-1, 1)
    corr_feat_top1    = (cov_f_T / (var_feat.sqrt() * std_T)).clamp(-1, 1)

    W = model.dec.weight.detach().cpu()  # [d_in, d_sae]
    stats = {
        "activation_rate": act_rate,
        "mean_when_active": mean_when_active,
        "corr_entropy": corr_feat_entropy,
        "corr_top1prob": corr_feat_top1,
    }
    return stats, W

def feature_card(j: int, stats, W, topk_experts: int = 5):
    act = stats["activation_rate"][j].item()
    mwa = stats["mean_when_active"][j].item()
    cH = stats["corr_entropy"][j].item()
    cT = stats["corr_top1prob"][j].item()
    col = W[:, j]
    # expert concentration
    w_abs = col.abs()
    top_idx = torch.topk(w_abs, k=min(topk_experts, w_abs.numel())).indices.tolist()
    top_list = [(i, col[i].item()) for i in top_idx]
    mass1 = w_abs[top_idx[0]] / (w_abs.sum().clamp_min(1e-8))
    mass12 = w_abs[top_idx[0:2]].sum() / (w_abs.sum().clamp_min(1e-8))
    return {
        "feature": j,
        "act_rate": act,
        "mean_when_active": mwa,
        "corr_entropy": cH,
        "corr_top1prob": cT,
        "top_experts": top_list,
        "mass_top1": mass1.item(),
        "mass_top2": mass12.item(),
    }

@torch.no_grad()
def ablate_effect(model: SAE, device: str, feat_j: int, n_examples: int = 4096, k: Optional[int] = None):
    """Mean change in reconstructed probs when zeroing feature j."""
    xs = []
    for xb in batch_iter(n_examples, batch_size=min(4096, n_examples)):
        xs.append(xb)
        if sum(x.shape[0] for x in xs) >= n_examples:
            break
    x = torch.cat(xs, dim=0)[:n_examples].to(device)
    pre = torch.relu(model.enc(x))
    if k is not None and k > 0:
        vals, _ = torch.topk(pre, k=min(k, pre.shape[1]), dim=1)
        kth = vals[:, -1].unsqueeze(1)
        mask = (pre >= kth).to(pre.dtype)
        codes = pre * mask
    else:
        codes = pre

    logits = model.dec(codes)
    p = torch.softmax(logits, dim=-1)

    codes_abl = codes.clone()
    codes_abl[:, feat_j] = 0.0
    logits_abl = model.dec(codes_abl)
    p_abl = torch.softmax(logits_abl, dim=-1)

    delta = (p_abl - p).mean(dim=0).cpu()  # average effect
    top_change = torch.topk(delta.abs(), k=5)
    return delta, [(i.item(), delta[i].item()) for i in top_change.indices]

def write_csv(rows: List[dict], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join(cfg.save_dir, "sae_final.pt"))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--k", type=int, default=8)   # match your training Top-K
    parser.add_argument("--cards", type=int, default=20)
    args = parser.parse_args()

    device = cfg.device
    model, d_in, d_sae = load_model(args.ckpt, device)

    stats, W = compute_feature_stats(model, device, n_examples=args.samples, topk=args.k)

    # Rank features by different notions
    act_rate = stats["activation_rate"]
    top_active = torch.topk(act_rate, k=min(args.cards, act_rate.numel()))
    by_purity = torch.topk((W.abs().max(dim=0).values / W.abs().sum(dim=0).clamp_min(1e-8)), k=min(args.cards, W.shape[1]))
    by_pair = torch.topk(  # mass on top-2
        torch.topk(W.abs(), k=2, dim=0).values.sum(dim=0) / W.abs().sum(dim=0).clamp_min(1e-8),
        k=min(args.cards, W.shape[1])
    )
    by_uncertainty = torch.topk(stats["corr_entropy"].abs(), k=min(args.cards, W.shape[1]))

    rows = []
    print("\n=== Top by activation rate ===")
    for j in top_active.indices.tolist():
        card = feature_card(j, stats, W)
        rows.append(card)
        print(card)

    print("\n=== Most single-expert (mass_top1) ===")
    for j in by_purity.indices.tolist():
        print(feature_card(j, stats, W))

    print("\n=== Most pairwise (mass_top2) ===")
    for j in by_pair.indices.tolist():
        print(feature_card(j, stats, W))

    print("\n=== Most (un)certainty-correlated (|corr_entropy|) ===")
    for j in by_uncertainty.indices.tolist():
        print(feature_card(j, stats, W))

    # Ablate a few interesting features
    print("\n=== Ablation examples ===")
    for j in top_active.indices[:5].tolist():
        delta, topchg = ablate_effect(model, device, j, n_examples=8192, k=args.k)
        print(f"feat {j} ablation top changes:", topchg)

    # Save all cards to CSV
    all_rows = [feature_card(j, stats, W) for j in range(W.shape[1])]
    out_csv = os.path.join(cfg.save_dir, "feature_cards.csv")
    write_csv(all_rows, out_csv)
    print(f"\nWrote {out_csv} ({len(all_rows)} rows)")

if __name__ == "__main__":
    main()
