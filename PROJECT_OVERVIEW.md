# Project Overview: SAE for GPT-OSS-20B MoE Router Analysis

This project trains Sparse Autoencoders (SAEs) on the router activations from GPT-OSS-20B (a 32-expert mixture-of-experts model) and provides tools to analyze expert specialization patterns.

## File Dependencies & Descriptions

### Core Training & Analysis

**sae.py** (standalone)
- Trains a Sparse Autoencoder on router log-probabilities from GPT-OSS-20B MoE layers
- Uses Top-K sparsification and KL divergence loss with L1 regularization
- Outputs: Trained SAE checkpoints in `sae_checkpoints_expert_router/`

**analyze.py** → depends on: `sae.py`
- Analyzes trained SAE models: computes reconstruction metrics, feature correlations with router properties
- Identifies top-activating experts and tokens for each SAE feature
- Outputs: CSV files with feature statistics and correlations

**analyze_text.py** → depends on: `sae.py`
- Shows top-activating prompt/generated text contexts for each SAE feature
- Fixes correlation computations from analyze.py and adds text-based feature interpretation
- Outputs: CSV files with text contexts ranked by feature activation strength

**linear_probe.py** (standalone)
- Fits PCA (with optional varimax rotation) on router log-probabilities as a linear baseline
- Provides component cards, top contexts, KL reconstruction metrics, and ablation analysis
- Outputs: PCA model checkpoints and feature analysis CSVs

**router_tools.py** (standalone)
- Utility functions for analyzing raw router distributions across layers
- Computes expert selection statistics, entropy, and top-p metrics per layer
- Used for baseline router behavior analysis without SAE

### Dashboard & Visualization

**export_expert_dashboard.py** (standalone)
- Exports router activation data to JSON format for visualization
- Tracks top-K contexts where each expert was selected most strongly
- Outputs: `expert_dashboard_layer21.json` (or similar)

**export_sae_features.py** → depends on: `sae.py`, `analyze_text.py`
- Exports per-token SAE feature activations to JSON for color-coded text highlighting
- Shows which parts of text activate specific SAE features most strongly
- Outputs: Feature-specific JSON files (e.g., `feature_42.json`)

**export_all_features.py** → depends on: `export_sae_features.py`
- Batch exports multiple SAE features at once for comparison
- Creates organized folder structure with all feature exports
- Outputs: Directory with multiple feature JSON files (e.g., `feature_exports/`)

**expert-dashboard/** (Next.js app) → depends on: `export_expert_dashboard.py`, `export_sae_features.py` (for data)
- React dashboard with two modes: Router view and Feature view
- Router view: Interactive exploration of expert statistics, probability distributions, and top contexts
- Feature view: Color-coded text highlighting based on SAE feature activation strength
- Features: search, filtering, sorting, CSV export, dual-mode visualization
- Run: `cd expert-dashboard && npm run dev`

### Utility

**main.py** (standalone)
- Placeholder entry point (currently just prints "Hello from sae!")

## Data Pipeline

```
HuggingFace Dataset (GPT-OSS-20B router activations)
    ↓
├─→ sae.py → trains SAE → checkpoints
│   ├─→ analyze.py → feature stats & correlations
│   └─→ analyze_text.py → text contexts per feature
│
├─→ linear_probe.py → PCA baseline → checkpoints & analysis
│
├─→ export_expert_dashboard.py → expert_dashboard_layer21.json
│   └─→ expert-dashboard/ (React app) → router visualization
│
├─→ export_sae_features.py → feature_N.json
│   └─→ expert-dashboard/ (React app) → feature activation visualization
│
└─→ router_tools.py → raw router statistics
```

## Key Outputs

- `sae_checkpoints_expert_router/`: SAE model checkpoints + TensorBoard logs
- `linear_probe_chkpts/`: PCA model checkpoints
- `expert_dashboard_layer21.json`: Expert router data for dashboard
- Various CSV files: Feature statistics, correlations, and text contexts

## Common Workflows

1. **Train SAE**: `uv run sae.py` (uses config in sae.py)
2. **Analyze features**: `uv run analyze_text.py --ckpt sae_checkpoints_*/sae_final.pt --feature 5`
3. **Linear baseline**: `uv run linear_probe.py fit --k 16 --samples 200000`
4. **Visualize experts**: `uv run export_expert_dashboard.py --layer 21 --out data.json` → load in dashboard
5. **Visualize SAE features**: `uv run export_sae_features.py --feature 42 --out feature_42.json` → load in dashboard
