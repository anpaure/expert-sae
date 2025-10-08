"""
export_expert_dashboard.py - Export Router Data for Dashboard Visualization

Exports router activation data to JSON format for the React dashboard.
Tracks top-K contexts where each expert was selected most strongly with statistics.

Dependencies: None (standalone)
Usage: uv run export_expert_dashboard.py --layer 21 --samples 120000 --keep 12 --out expert_dashboard_layer21.json
"""
import os, io, json, zipfile, pickle, argparse
from typing import Dict, Any, Iterator, List, Tuple, Optional
import heapq
import torch
import torch.nn.functional as F
from huggingface_hub import list_repo_files, hf_hub_download

HF_REPO = "AmanPriyanshu/GPT-OSS-20B-MoE-expert-activations"
INCLUDE = ["gpqa_diamond.zip"]  # add more later (mmlu, mmlu_pro, ...)

def _layer_key(l1: int) -> str: return f"layer_{l1-1}"

def _infer_n(obj: Dict[str, Any], key: str) -> int:
    rd = obj.get("router_data"); 
    if not rd: return 0
    for tb in rd:
        for _tok, layers in tb.items():
            if key in layers:
                fs = layers[key].get("full_scores")
                if fs is not None: return len(fs)
    return 0

def _iter_tokens(layer_1idx: int, max_tokens: Optional[int]) -> Iterator[Tuple[int, torch.Tensor, str, str]]:
    files = list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    zips = [p for p in files if p.lower().endswith(".zip")]
    if INCLUDE:
        zips = [p for p in zips if p in set(INCLUDE)]
    lk = _layer_key(layer_1idx)
    seen = 0
    n_hint = None

    def handle(obj):
        nonlocal n_hint, seen
        pr = obj.get("prompt",""); gen = obj.get("generated_only","")
        rd = obj.get("router_data"); 
        if not rd: return
        if not n_hint:
            n_hint = _infer_n(obj, lk)
            if not n_hint: return
        for tb in rd:
            for _tok, layers in tb.items():
                e = layers.get(lk)
                if not e: continue
                fs = e.get("full_scores")
                if fs is None: continue
                p = torch.tensor(fs, dtype=torch.float32)  # probs
                x = torch.log(p.clamp_min(1e-12))          # log-probs for entropy calc
                yield (n_hint, p, pr, gen)
                seen += 1
                if max_tokens and seen >= max_tokens: return

    for rel in zips:
        local = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=rel)
        with zipfile.ZipFile(local, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"): continue
                if name.lower().endswith((".pkl",".pickle")):
                    with zf.open(name,"r") as fh:
                        bio = io.BytesIO(fh.read())
                        try:
                            obj = pickle.load(bio); yield from handle(obj)
                        except Exception:
                            bio.seek(0)
                            while True:
                                try: obj = pickle.load(bio)
                                except EOFError: break
                                except Exception: break
                                yield from handle(obj)
                elif name.lower().endswith(".json"):
                    with zf.open(name,"r") as fh:
                        try: obj = json.load(fh); yield from handle(obj)
                        except Exception: continue
        if max_tokens and seen >= max_tokens: return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--samples", type=int, default=120000)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    layer = args.layer
    n_experts = None
    # heaps[e] keeps top-K contexts by p[e]
    heaps: Dict[int, List[Tuple[float, Tuple[str,str]]]] = {}
    count = [0]*32
    H_sum = [0.0]*32
    top_sum = [0.0]*32

    for n_experts, p, pr, gen in _iter_tokens(layer, args.samples):
        e = int(p.argmax().item())
        pe = float(p[e].item())
        H = float((-(p * (p.clamp_min(1e-12).log()))).sum().item())
        count[e] += 1
        H_sum[e] += H
        top_sum[e] += pe
        heaps.setdefault(e, [])
        item = (pe, (pr, gen))
        if len(heaps[e]) < args.keep:
            heapq.heappush(heaps[e], item)
        else:
            if pe > heaps[e][0][0]:
                heapq.heapreplace(heaps[e], item)

    if n_experts is None:
        raise SystemExit("No data found.")

    experts = []
    for e in range(n_experts):
        top = sorted(heaps.get(e, []), key=lambda t: -t[0])
        hits = [{"p": float(s), "prompt": pr[:4000], "generated": gen[:4000]} for s,(pr,gen) in top]
        c = count[e] or 1
        experts.append({
            "id": e,
            "stats": {
                "count": count[e],
                "meanTopP": top_sum[e]/c if count[e] else 0.0,
                "meanEntropy": H_sum[e]/c if count[e] else 0.0,
            },
            "top": hits
        })

    out = {
        "meta": {"layer": layer, "n_experts": n_experts, "keep": args.keep, "samples": args.samples},
        "experts": experts,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {args.out} (layer {layer}, {n_experts} experts)")

if __name__ == "__main__":
    main()