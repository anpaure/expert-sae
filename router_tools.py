"""
router_tools.py - Raw Router Distribution Analysis Utilities

Utility functions for analyzing router distributions across layers without SAE.
Computes expert selection statistics, entropy, top-p metrics, and baseline router behavior.

Dependencies: None (standalone)
Usage: Import functions or run as script with custom analysis
"""
import os, io, json, zipfile, pickle, argparse, math, heapq
from typing import Dict, Any, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import list_repo_files, hf_hub_download

# ---------- dataset config (matches your setup) ----------
HF_REPO = "AmanPriyanshu/GPT-OSS-20B-MoE-expert-activations"
INCLUDE = ["gpqa_diamond.zip"]        # expand later (mmlu, etc.)
LAYERS_1IDX = list(range(1, 25))      # we’ll filter in code per cmd

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

def _layer_keys(subset: Optional[List[int]] = None) -> List[str]:
    if subset is None:
        subset = LAYERS_1IDX
    return [f"layer_{i-1}" for i in subset]

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
                    if sel: return max(sel)+1
    return 0

def _logits(entry: Dict[str, Any], n: int) -> Optional[torch.Tensor]:
    scores = entry.get("full_scores", None)
    if scores is None: return None
    p = torch.tensor(scores, dtype=torch.float32)
    return p.clamp_min(1e-12).log()

def iter_records(max_tokens: Optional[int],
                 need_layers: List[int]) -> Iterator[Tuple[int, Dict[int, torch.Tensor], str, str]]:
    """
    Yields per-token dict of {layer_1idx: log-probs}, plus prompt/gen.
    """
    files = list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    zips = [p for p in files if p.lower().endswith(".zip")]
    if INCLUDE:
        zips = [p for p in zips if p in set(INCLUDE)]
    layer_keys = _layer_keys(need_layers)

    n_seen = 0
    n_hint = None

    def handle_obj(obj):
        nonlocal n_hint, n_seen
        prompt = obj.get("prompt","")
        gen = obj.get("generated_only","")
        rd = obj.get("router_data", None)
        if not rd:
            return
        if not n_hint:
            n_hint = _infer_n_experts(obj, layer_keys)
            if not n_hint:
                return
        for tb in rd:
            for _tok_name, layers in tb.items():
                x_by_layer = {}
                for lk in layer_keys:
                    if lk not in layers:
                        break
                    x = _logits(layers[lk], n_hint)
                    if x is None:
                        break
                    l1 = int(lk.split("_")[1]) + 1  # to 1-indexed
                    x_by_layer[l1] = x
                else:
                    # only hit if we didn't break: we have all requested layers
                    yield (n_hint, x_by_layer, prompt, gen)
                    n_seen += 1
                    if max_tokens and n_seen >= max_tokens:
                        return

    for rel in zips:
        local = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=rel)
        with zipfile.ZipFile(local, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                if name.lower().endswith((".pkl", ".pickle")):
                    with zf.open(name, "r") as fh:
                        bio = io.BytesIO(fh.read())
                        try:
                            obj = pickle.load(bio)
                            # BUGFIX: forward the inner generator's yields
                            yield from handle_obj(obj)
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
                            obj = json.load(fh)
                            yield from handle_obj(obj)
                        except Exception:
                            continue
        # stop once we’ve hit the global cap
        if max_tokens and n_seen >= max_tokens:
            return

# ---------- A) early prediction ----------
class SoftmaxProbe(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.W = nn.Linear(d_in, n_classes, bias=True)
    def forward(self, x):  # x: [B,d]
        return self.W(x)

@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())

def cmd_early_predict(args):
    target = args.target_layer
    layers = list(range(1, target+1))  # predict target from each <= target
    data_it = iter_records(max_tokens=args.samples, need_layers=layers)
    # stream into small buffers per source layer to train quickly
    per_layer_X, per_layer_Y = {l: [] for l in layers}, {l: [] for l in layers}
    n_experts = None
    for n_experts, x_by_l, *_ in data_it:
        y = int(torch.exp(x_by_l[target]).argmax().item())
        for l in layers:
            per_layer_X[l].append(x_by_l[l])
            per_layer_Y[l].append(y)
    if n_experts is None:
        print("No data found."); return
    print(f"Collected examples with n_experts={n_experts}")
    # train small probes
    for l in layers:
        X = torch.stack(per_layer_X[l]).to(DEVICE)
        y = torch.tensor(per_layer_Y[l], dtype=torch.long, device=DEVICE)
        model = SoftmaxProbe(d_in=n_experts, n_classes=n_experts).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        B = min(8192, X.size(0))
        for _ in range(args.epochs):
            for i in range(0, X.size(0), B):
                xb = X[i:i+B]; yb = y[i:i+B]
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward(); opt.step()
        with torch.no_grad():
            logits = model(X)
            acc = _accuracy(logits, y)
        print(f"Layer {l:2d} → Layer {target:2d} top1 acc: {acc:.3f}")

# ---------- B) expert dashboard ----------
def cmd_expert_dashboard(args):
    layers = [args.layer]
    it = iter_records(max_tokens=args.samples, need_layers=layers)
    n_experts = None
    # keep top contexts per expert by probability
    heaps = {}
    for n_experts, x_by_l, pr, gen in it:
        p = F.softmax(x_by_l[args.layer], dim=-1)
        score, e = float(p.max().item()), int(p.argmax().item())
        heaps.setdefault(e, [])
        item = (score, (pr, gen))
        if len(heaps[e]) < args.keep:
            heapq.heappush(heaps[e], item)
        else:
            if score > heaps[e][0][0]:
                heapq.heapreplace(heaps[e], item)
    if n_experts is None:
        print("No data found."); return
    for e in range(n_experts):
        if e not in heaps: continue
        top = sorted(heaps[e], key=lambda t: -t[0])
        print(f"\n=== Expert {e} | top {len(top)} contexts ===")
        for s,(pr,gen) in top:
            print(f"[p={s:.3f}] PROMPT: {pr[:240]}")
            print(f"         GENERATED: {gen[:240]}\n")

# ---------- C) soft steering sim ----------
def load_probe(path: str):
    obj = torch.load(path, map_location="cpu")
    return obj["mean"], obj["U"]

def steer_component(x: torch.Tensor, mean: torch.Tensor, U: torch.Tensor, j: int, alpha: float):
    # x: log-probs (unnormalized), apply delta = alpha * U[:,j]
    return F.softmax(x + alpha * U[:, j], dim=-1)

def min_kl_to_target(x: torch.Tensor, target_e: int, lam: float = 10.0, steps: int = 50, lr: float = 0.5):
    """
    Find small delta that increases q[target_e] while penalizing KL(q || softmax(x)).
    Returns q_new, delta.
    """
    d = x.numel()
    delta = torch.zeros(d, requires_grad=True)
    opt = torch.optim.SGD([delta], lr=lr)
    for _ in range(steps):
        q = F.softmax(x + delta, dim=-1)
        # maximize q[target]  == minimize -log q[target]
        loss_target = -torch.log(q[target_e] + 1e-12)
        # keep close to original routing (reverse KL)
        q0 = F.softmax(x.detach(), dim=-1)
        loss_kl = F.kl_div(q.log(), q0, reduction="batchmean", log_target=True)
        loss = loss_target + lam * loss_kl
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        q = F.softmax(x + delta, dim=-1)
    return q, delta.detach()

def cmd_steer_sim(args):
    layers = [args.layer]
    it = iter_records(max_tokens=args.samples, need_layers=layers)
    n_experts = None
    flips = 0; total = 0
    ent_change = 0.0
    if args.probe is not None:
        mean,U = load_probe(args.probe)
        U = U.to(torch.float32)
    for n_experts, x_by_l, *_ in it:
        x = x_by_l[args.layer]
        q0 = F.softmax(x, dim=-1)
        e0 = int(q0.argmax().item())
        if args.mode == "component":
            q1 = steer_component(x, mean=None, U=U, j=args.feature, alpha=args.alpha)
        else:
            q1, _ = min_kl_to_target(x, target_e=args.target, lam=args.lam, steps=args.steps, lr=args.lr)
        e1 = int(q1.argmax().item())
        flips += int(e1 != e0)
        total += 1
        H = lambda q: float((-(q * (q.clamp_min(1e-12).log()))).sum().item())
        ent_change += (H(q1) - H(q0))
    if total == 0:
        print("No data found."); return
    print(f"Flip-rate: {flips/total:.3f} over {total} tokens | mean Δentropy: {ent_change/total:+.4f}")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("early-predict")
    q.add_argument("--samples", type=int, default=200000)
    q.add_argument("--epochs", type=int, default=2)
    q.add_argument("--target-layer", type=int, default=24)

    d = sub.add_parser("expert-dashboard")
    d.add_argument("--layer", type=int, default=21)
    d.add_argument("--samples", type=int, default=120000)
    d.add_argument("--keep", type=int, default=8)

    s = sub.add_parser("steer-sim")
    s.add_argument("--layer", type=int, default=21)
    s.add_argument("--samples", type=int, default=20000)
    s.add_argument("--mode", choices=["component","target"], default="component")
    s.add_argument("--probe", type=str, default=None, help="required for mode=component")
    s.add_argument("--feature", type=int, default=0)
    s.add_argument("--alpha", type=float, default=1.0)
    s.add_argument("--target", type=int, default=0)
    s.add_argument("--lam", type=float, default=10.0)
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--lr", type=float, default=0.5)

    args = p.parse_args()
    print(f"Device: {DEVICE}")

    if args.cmd == "early-predict":
        cmd_early_predict(args)
    elif args.cmd == "expert-dashboard":
        cmd_expert_dashboard(args)
    elif args.cmd == "steer-sim":
        if args.mode == "component" and args.probe is None:
            raise SystemExit("mode=component requires --probe (your varimax probe .pt)")
        cmd_steer_sim(args)

if __name__ == "__main__":
    main()
