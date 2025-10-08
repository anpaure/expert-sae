# sae_v3.py - Sparse Autoencoder Training for GPT-OSS-20B MoE Router Activations
# Patched version: adds JumpReLU gating (with optional Top-K cap), k-anneal,
# lifetime balance, density targeting, dead-feature resurrection, orthogonality, better logs.
#
# Usage: uv run sae_v3.py
# (adjust Config below as needed)

import os, io, json, zipfile, pickle, random
from dataclasses import dataclass, asdict, field
from typing import Iterator, Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm
import wandb


# --------------------
# Config
# --------------------
@dataclass
class Config:
    # Data
    hf_dataset_repo: str = "AmanPriyanshu/GPT-OSS-20B-MoE-expert-activations"
    layers_1_indexed: List[int] = field(default_factory=lambda: [21])  # 1-indexed layers
    all_layers: bool = False

    # Representation:
    #  - "logits": log-probs from full router distribution (recommended)
    #  - "probs":  raw probability vectors
    #  - "selected": K-hot selected experts (optionally prob-weighted)
    input_mode: str = "logits"
    use_selected: bool = False
    weight_by_prob: bool = True

    # Normalization
    normalize_inputs: bool = True         # ignored for logits/probs (we re-normalize inside loss)
    normalize_per_layer: bool = False

    # SAE size / sparsity
    d_sae_multiple: int = 8               # e.g., 32 experts -> 256 features

    # === Gating mode ===
    # "topk": classic ReLU + hard top-k
    # "jumprelu": JumpReLU (learned thresholds), no cap
    # "jumprelu_cap": JumpReLU + optional hard top-k cap (recommended starting point)
    gate_mode: str = "jumprelu_cap"

    # Top-k schedule (used if gate_mode in {"topk","jumprelu_cap"})
    topk_start: int = 16
    topk_end: int = 8
    topk_anneal_steps: int = 800

    # L1 on codes (still useful with JumpReLU to sharpen features)
    l1_coeff: float = 2e-3
    l1_coeff_max: float = 2e-2
    l1_warmup_steps: int = 600

    # Global density targeting (aim average active fraction to target)
    # If None, target derived from current_k / d_sae when top-k is enabled; else fixed override
    density_target_override: Optional[float] = None
    density_coeff: float = 5e-2
    density_beta: float = 0.98  # EMA for global density

    # Lifetime balance (per-feature usage equalization)
    balance_coeff: float = 0.20
    balance_beta: float = 0.95

    # Orthogonality on decoder columns
    orth_coeff: float = 1e-4

    # Dead feature resurrection
    resurrect_every: int = 200
    resurrect_threshold: float = 2e-3
    max_resurrect_per_check: int = 64

    # Optim / schedule
    batch_size: int = 4096
    steps: int = 3_000
    lr: float = 5e-4
    grad_clip: float = 1.0

    # Device / perf
    device: str = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available()
                   else "cpu")
    compile_model: bool = True            # disabled on MPS below
    use_autocast: bool = True             # disabled on MPS below

    # Logging / checkpoints
    seed: int = 42
    save_dir: str = "./sae_checkpoints_expert_router_v3"
    log_every: int = 50
    eval_every: int = 500

    # Dataset slicing
    max_tokens: Optional[int] = 200_000
    include_files: Optional[List[str]] = field(default_factory=lambda: ["gpqa_diamond.zip"])

    # JumpReLU init
    jumprelu_tau_init: float = 0.0        # initial threshold
    jumprelu_bias_init: float = 0.0       # optional encoder bias init for stability


cfg = Config()

random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
os.makedirs(cfg.save_dir, exist_ok=True)


def _device_is_cuda() -> bool:
    return cfg.device.startswith("cuda")


def _device_is_mps() -> bool:
    return cfg.device == "mps"


PIN_MEMORY = _device_is_cuda()     # CUDA only
NON_BLOCKING = _device_is_cuda()
USE_COMPILE = cfg.compile_model and _device_is_cuda()   # disable on MPS
USE_AMP = cfg.use_autocast and _device_is_cuda()        # disable on MPS

if _device_is_cuda():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# --------------------
# Data loading utils
# --------------------
def _vectorize_entry(entry: Dict[str, Any],
                     n_experts: int,
                     mode: str,
                     weight_by_prob: bool) -> Optional[torch.Tensor]:
    eps = 1e-6
    if mode == "selected":
        sel = entry.get("selected_experts", None)
        if sel is None:
            return None
        probs = entry.get("selected_probs", [1.0] * len(sel))
        x = torch.zeros(n_experts, dtype=torch.float32)
        for e, p in zip(sel, probs):
            e = int(e)
            if 0 <= e < n_experts:
                x[e] = float(p) if weight_by_prob else 1.0
        return x

    scores = entry.get("full_scores", None)  # probabilities
    if scores is None:
        return None
    p = torch.tensor(scores, dtype=torch.float32)
    if mode == "probs":
        return p
    elif mode == "logits":
        return torch.log(p.clamp_min(eps))  # log-probs
    else:
        raise ValueError(f"Unknown input_mode={mode}")


def _infer_n_experts(obj: Dict[str, Any], layer_keys: List[str]) -> int:
    rd = obj.get("router_data", None)
    if not rd:
        return 0
    for tb in rd:
        for _tok, layers in tb.items():
            for lk in layer_keys:
                if lk in layers:
                    e = layers[lk]
                    fs = e.get("full_scores", None)
                    if fs is not None:
                        return len(fs)
                    sel = e.get("selected_experts", [])
                    if sel:
                        return max(sel) + 1
    return 0


def _iter_vectors_from_zip(zip_path: str,
                           layer_keys: List[str],
                           n_experts_hint: Optional[int]) -> Iterator[Tuple[int, torch.Tensor]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith('/'):
                continue

            def handle_obj(obj):
                nonlocal n_experts_hint
                if not n_experts_hint:
                    n_experts_hint = _infer_n_experts(obj, layer_keys)
                    if not n_experts_hint:
                        return
                rd = obj.get("router_data", None)
                if not rd:
                    return
                for tb in rd:
                    for _tok, layers in tb.items():
                        for lk in layer_keys:
                            if lk not in layers:
                                continue
                            vec = _vectorize_entry(layers[lk], n_experts_hint, cfg.input_mode, cfg.weight_by_prob)
                            if vec is not None:
                                z = int(lk.split("_")[1]) if "_" in lk else 0
                                yield (z + 1, vec)

            if name.lower().endswith((".pkl", ".pickle")):
                with zf.open(name, "r") as fh:
                    bio = io.BytesIO(fh.read())
                    try:
                        obj = pickle.load(bio)
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


def _layer_keys() -> List[str]:
    if cfg.all_layers:
        return [f"layer_{i}" for i in range(64)]
    else:
        return [f"layer_{i - 1}" for i in cfg.layers_1_indexed]


def iter_all_vectors(max_tokens: Optional[int] = None) -> Iterator[Tuple[int, torch.Tensor]]:
    files = list_repo_files(repo_id=cfg.hf_dataset_repo, repo_type="dataset")
    zips = [p for p in files if p.lower().endswith(".zip")]
    if cfg.include_files:
        zips = [p for p in zips if p in set(cfg.include_files)]

    seen = 0
    layer_keys = _layer_keys()
    n_experts_hint: Optional[int] = None

    for rel in zips:
        local = hf_hub_download(repo_id=cfg.hf_dataset_repo, repo_type="dataset", filename=rel)
        for l1, vec in _iter_vectors_from_zip(local, layer_keys, n_experts_hint):
            if n_experts_hint is None:
                n_experts_hint = vec.numel()
            yield (l1, vec)
            seen += 1
            if max_tokens is not None and seen >= max_tokens:
                return


# --------------------
# Dataset
# --------------------
class ExpertRouterIterable(IterableDataset):
    def __init__(self, max_tokens: Optional[int], normalize_inputs: bool, normalize_per_layer: bool):
        super().__init__()
        self.max_tokens = max_tokens
        self.normalize_inputs = normalize_inputs
        self.normalize_per_layer = normalize_per_layer
        self._mean_global: Optional[torch.Tensor] = None
        self._mean_per_layer: Dict[int, torch.Tensor] = {}
        self._n_experts: Optional[int] = None

    def compute_running_mean(self, sample_count: int = 200_000):
        cnt_global = 0
        mean_global = None
        counts_per_layer: Dict[int, int] = {}
        sums_per_layer: Dict[int, torch.Tensor] = {}

        for _i, (l1, x) in enumerate(iter_all_vectors(max_tokens=sample_count)):
            if self._n_experts is None:
                self._n_experts = x.numel()
            if mean_global is None:
                mean_global = torch.zeros_like(x, dtype=torch.float64)
            mean_global += x.to(torch.float64)
            cnt_global += 1
            if self.normalize_per_layer:
                sums_per_layer.setdefault(l1, torch.zeros_like(x, dtype=torch.float64))
                counts_per_layer[l1] = counts_per_layer.get(l1, 0) + 1
                sums_per_layer[l1] += x.to(torch.float64)

        if cnt_global > 0:
            self._mean_global = (mean_global / cnt_global).to(torch.float32)
        if self.normalize_per_layer:
            for l1, s in sums_per_layer.items():
                self._mean_per_layer[l1] = (s / counts_per_layer[l1]).to(torch.float32)

    def __iter__(self):
        for l1, x in iter_all_vectors(max_tokens=self.max_tokens):
            if self._n_experts is None:
                self._n_experts = x.numel()
            if self.normalize_inputs:
                if self.normalize_per_layer and l1 in self._mean_per_layer:
                    yield x - self._mean_per_layer[l1]
                elif self._mean_global is not None:
                    yield x - self._mean_global
                else:
                    yield x
            else:
                yield x

    @property
    def n_experts(self) -> int:
        if self._n_experts is None:
            for _l1, x in iter_all_vectors(max_tokens=1):
                self._n_experts = x.numel()
                break
        if self._n_experts is None:
            raise ValueError("Could not infer n_experts from dataset")
        return int(self._n_experts)


# --------------------
# JumpReLU gate
# --------------------
class JumpReLU(nn.Module):
    """
    y = relu(x - tau), with learnable per-feature thresholds tau.
    Returns y and binary mask (pre > tau).
    """
    def __init__(self, d_sae: int, tau_init: float = 0.0):
        super().__init__()
        self.tau = nn.Parameter(torch.full((d_sae,), float(tau_init)))

    def forward(self, pre: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pre: [B, d_sae]
        y = F.relu(pre - self.tau)
        mask = (pre > self.tau).to(pre.dtype)
        return y, mask


# --------------------
# Model
# --------------------
class SAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, gate_mode: str = "jumprelu_cap", tie_dec: bool = False,
                 tau_init: float = 0.0, bias_init: float = 0.0):
        super().__init__()
        self.enc = nn.Linear(d_in, d_sae, bias=True)
        self.dec = nn.Linear(d_sae, d_in, bias=False)
        self.tie_dec = tie_dec
        self.gate_mode = gate_mode

        # init
        nn.init.kaiming_uniform_(self.enc.weight, a=0.0)
        with torch.no_grad():
            self.enc.bias.fill_(bias_init)
        nn.init.zeros_(self.dec.weight)

        # gating
        if gate_mode in ("jumprelu", "jumprelu_cap"):
            self.gate = JumpReLU(d_sae, tau_init=tau_init)
        else:
            self.gate = None  # topk (vanilla)

    def forward(self, x: torch.Tensor, k_cap: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (recon_logits, codes, mask)
        mask is the binary activation mask *after* the final gating step.
        """
        pre = self.enc(x)  # [B, d_sae]

        if self.gate_mode == "topk":
            codes = F.relu(pre)
            mask = (codes > 0).to(codes.dtype)
            if (k_cap is not None) and (k_cap > 0):
                k = min(k_cap, codes.shape[1])
                vals, _ = torch.topk(codes, k=k, dim=1)
                kth = vals[:, -1].unsqueeze(1)
                mask_k = (codes >= kth).to(codes.dtype)
                codes = codes * mask_k
                mask = mask * mask_k

        elif self.gate_mode == "jumprelu":
            codes, mask = self.gate(pre)

        elif self.gate_mode == "jumprelu_cap":
            codes, mask = self.gate(pre)
            if (k_cap is not None) and (k_cap > 0):
                k = min(k_cap, codes.shape[1])
                vals, _ = torch.topk(codes, k=k, dim=1)
                kth = vals[:, -1].unsqueeze(1)
                mask_k = (codes >= kth).to(codes.dtype)
                codes = codes * mask_k
                mask = mask * mask_k

        else:
            raise ValueError(f"Unknown gate_mode={self.gate_mode}")

        recon = self.dec(codes) if not self.tie_dec else F.linear(codes, self.enc.weight.t(), None)
        return recon, codes, mask


# --------------------
# Collate (top-level to avoid pickling issues)
# --------------------
def stack_collate(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


# --------------------
# Schedules & losses
# --------------------
def _current_k(step: int) -> int:
    if cfg.gate_mode not in ("topk", "jumprelu_cap"):
        return 0
    if step >= cfg.topk_anneal_steps:
        return cfg.topk_end
    return int(round(cfg.topk_start + (cfg.topk_end - cfg.topk_start) * (step / max(1, cfg.topk_anneal_steps))))


def _orthogonality_loss(W: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    C = W / (W.norm(dim=0, keepdim=True) + eps)
    G = C.t() @ C
    I = torch.eye(G.shape[0], device=G.device)
    return ((G - I) ** 2).mean()


# --------------------
# Training
# --------------------
def train():
    # IMPORTANT: for logits/probs we disable input normalization to keep a valid distribution
    effective_norm = cfg.normalize_inputs and (cfg.input_mode == "selected")
    if cfg.input_mode in ("logits", "probs") and cfg.normalize_inputs:
        print("Note: disabling input normalization for logits/probs to avoid invalid KL targets.")

    dataset = ExpertRouterIterable(
        max_tokens=cfg.max_tokens,
        normalize_inputs=effective_norm,
        normalize_per_layer=cfg.normalize_per_layer
    )

    print(f"Device: {cfg.device}")
    if effective_norm:
        print("Computing running mean for normalization...")
        dataset.compute_running_mean(sample_count=min(200_000, cfg.max_tokens or 200_000))
        if dataset._mean_global is not None:
            torch.save(dataset._mean_global, os.path.join(cfg.save_dir, "input_running_mean_global.pt"))
        if cfg.normalize_per_layer and dataset._mean_per_layer:
            torch.save(dataset._mean_per_layer, os.path.join(cfg.save_dir, "input_running_mean_per_layer.pt"))
    else:
        print("Skipping input normalization.")

    d_in = dataset.n_experts
    d_sae = cfg.d_sae_multiple * d_in
    print(f"Detected n_experts (d_in) = {d_in}; using d_sae = {d_sae}")

    model = SAE(
        d_in=d_in,
        d_sae=d_sae,
        gate_mode=cfg.gate_mode,
        tie_dec=False,
        tau_init=cfg.jumprelu_tau_init,
        bias_init=cfg.jumprelu_bias_init,
    ).to(cfg.device)

    if USE_COMPILE:
        try:
            model = torch.compile(model)
            print("Compiled model with torch.compile().")
        except Exception as e:
            print(f"torch.compile not used: {e}")
    else:
        if _device_is_mps():
            print("Skipping torch.compile on MPS for numerical stability.")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)

    # Initialize wandb
    wandb.init(
        project="sae-gpt-oss-20b",
        name=f"sae_v3_{cfg.gate_mode}_k{cfg.topk_start}to{cfg.topk_end}",
        config=asdict(cfg),
        tags=[cfg.gate_mode, f"d_sae_{d_sae}", f"layer_{cfg.layers_1_indexed[0]}"]
    )
    print(f"W&B: {wandb.run.url}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=stack_collate,
        pin_memory=PIN_MEMORY,
        num_workers=0,          # IterableDataset + macOS: keep 0 to avoid duplication/pickling issues
        drop_last=False
    )

    step = 0
    recon_key = "kld" if cfg.input_mode in ("logits", "probs") else "bce"
    running: Dict[str, float] = {"loss": 0.0, "l1": 0.0, "sparsity": 0.0, recon_key: 0.0, "n": 0}

    pbar = tqdm(total=cfg.steps, desc="Training SAE")
    data_iter = iter(loader)

    # AMP config
    amp_device = "cuda" if _device_is_cuda() else ("mps" if _device_is_mps() else "cpu")
    amp_dtype = torch.bfloat16 if _device_is_cuda() else torch.float32  # keep float32 on MPS/CPU

    # Lifetime balance / density tracking
    ema_usage = torch.zeros(d_sae, device=cfg.device)     # per-feature usage
    ema_density = torch.tensor(0.0, device=cfg.device)    # global average usage

    while step < cfg.steps:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)

        x = x.to(cfg.device, non_blocking=NON_BLOCKING)
        opt.zero_grad(set_to_none=True)

        k_now = _current_k(step)

        # Forward
        ctx = torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=USE_AMP)
        with ctx:
            recon_logits, codes, mask = model(x, k_cap=k_now if k_now > 0 else None)

        # ----- Reconstruction loss -----
        if cfg.input_mode == "selected":
            loss_recon = F.binary_cross_entropy_with_logits(recon_logits.float(), x.float())
        elif cfg.input_mode in ("probs", "logits"):
            # Build normalized log-target to avoid log(0) and zero-sum issues
            if cfg.input_mode == "logits":
                log_p = x.float()  # already log-probs
            else:  # probs
                log_p = x.clamp_min(1e-12).float().log()
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)  # normalize
            log_q = F.log_softmax(recon_logits.float(), dim=-1)
            loss_recon = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
        else:
            raise ValueError(f"Bad input_mode={cfg.input_mode}")

        # L1 on codes with warmup
        l1 = codes.float().abs().mean()
        ramp = min(1.0, step / max(1, cfg.l1_warmup_steps))
        l1_coeff_now = cfg.l1_coeff + ramp * (cfg.l1_coeff_max - cfg.l1_coeff)

        # Lifetime balance (per-feature)
        batch_usage = mask.float().mean(dim=0)  # [d_sae] fraction active in batch
        ema_usage.mul_(cfg.balance_beta).add_((1.0 - cfg.balance_beta) * batch_usage)

        # Density target (global)
        # If override provided, use it; else if we have topk schedule, derive k/d_sae; else keep current ema
        target_density = (
            float(cfg.density_target_override)
            if cfg.density_target_override is not None
            else (float(k_now) / float(d_sae) if k_now > 0 else ema_density.item())
        )
        current_density = mask.float().mean()  # scalar
        ema_density.mul_(cfg.density_beta).add_((1.0 - cfg.density_beta) * current_density)

        # Losses
        balance_loss = cfg.balance_coeff * ((ema_usage - target_density) ** 2).mean()
        density_loss = cfg.density_coeff * (ema_density - target_density) ** 2
        orth_loss = cfg.orth_coeff * _orthogonality_loss(model.dec.weight)

        loss = loss_recon + l1_coeff_now * l1 + balance_loss + density_loss + orth_loss
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Dead feature resurrection
        if (cfg.resurrect_every > 0) and (step % cfg.resurrect_every == 0) and step > 0:
            dead = (ema_usage < cfg.resurrect_threshold).nonzero(as_tuple=False).squeeze(-1)
            if dead.numel():
                n_fix = min(dead.numel(), cfg.max_resurrect_per_check)
                pick = dead[:n_fix]
                with torch.no_grad():
                    Wenc = model.enc.weight
                    benc = model.enc.bias
                    Wdec = model.dec.weight
                    Wenc[pick, :].normal_(0, 0.05)
                    benc[pick].fill_(0.01)
                    Wdec[:, pick].zero_()
                    # If JumpReLU, slightly lower the thresholds to encourage firing
                    if isinstance(model.gate, JumpReLU):
                        model.gate.tau.data[pick] = torch.full_like(model.gate.tau.data[pick], cfg.jumprelu_tau_init)
                ema_usage[pick] = max(ema_usage.mean().item(), target_density) * 0.5

        with torch.no_grad():
            sparsity = (codes > 0).float().mean().item()

        # Stats
        running[recon_key] += float(loss_recon.item())
        running["l1"] += float(l1.item())
        running["sparsity"] += sparsity
        running["loss"] += float(loss.item())
        running["n"] += 1

        step += 1
        pbar.update(1)

        if _device_is_mps() and (step % 200 == 0):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        if step % cfg.log_every == 0:
            n = running["n"]
            avg_recon = running[recon_key] / n
            avg_l1 = running["l1"] / n
            avg_sparsity = running["sparsity"] / n
            avg_loss = running["loss"] / n

            u = ema_usage.detach()
            H = float((u ** 2).sum().item())
            H_uniform = 1.0 / d_sae
            conc = H / H_uniform
            u_sorted = torch.sort(u).values
            denom = u_sorted.sum().clamp_min(1e-8)
            # Simple Gini proxy
            gini = 1.0 - 2.0 * ((u_sorted.cumsum(0) / denom) * (1.0 / d_sae)).sum().item()

            # W&B logging
            wandb.log({
                f"loss/{recon_key}": avg_recon,
                "loss/l1": avg_l1,
                "loss/total": avg_loss,
                "loss/l1_coeff_now": l1_coeff_now,
                "loss/orth": orth_loss.item(),
                "loss/balance": balance_loss.item(),
                "loss/density": density_loss.item(),
                "metrics/sparsity_fraction": avg_sparsity,
                "metrics/k_now": k_now,
                "metrics/ema_density": ema_density.item(),
                "metrics/target_density": target_density,
                "metrics/util_ema_min": float(u.min().item()),
                "metrics/util_ema_median": float(u.median().item()),
                "metrics/util_ema_max": float(u.max().item()),
                "metrics/util_concentration": float(conc),
                "metrics/util_gini": float(gini),
            }, step=step)

            print(
                f"[step {step}] loss={avg_loss:.5f} {recon_key}={avg_recon:.5f} l1={avg_l1:.5f} "
                f"mode={cfg.gate_mode} k={k_now} sparsity={avg_sparsity:.4f} "
                f"density(ema/target)={ema_density.item():.4f}/{target_density:.4f} "
                f"util[min/med/max]={float(u.min()):.4f}/{float(u.median()):.4f}/{float(u.max()):.4f} "
                f"conc={conc:.2f} gini={gini:.2f}"
            )

            running = {"loss": 0.0, "l1": 0.0, "sparsity": 0.0, recon_key: 0.0, "n": 0}

        if step % cfg.eval_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "d_in": d_in,
                    "d_sae": d_sae,
                    "gate_mode": cfg.gate_mode,
                },
                os.path.join(cfg.save_dir, f"sae_step_{step}.pt"),
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "d_in": d_in,
            "d_sae": d_sae,
            "gate_mode": cfg.gate_mode,
        },
        os.path.join(cfg.save_dir, "sae_final.pt"),
    )
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    train()
