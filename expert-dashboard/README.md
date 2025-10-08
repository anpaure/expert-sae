# Expert Router & SAE Feature Dashboard

Interactive dashboard for visualizing GPT-OSS-20B MoE router behavior and SAE feature activations.

## Features

### Router View
- Visualize expert selection patterns across 32 experts
- Interactive statistics: activation count, mean top-p, mean entropy
- Probability distribution histograms per expert
- Search and filter through top activation contexts
- Sort by count or mean top-p
- Export filtered results to CSV

### Feature View
- Color-coded text highlighting based on SAE feature activation strength
- Red = high activation, Blue = low activation
- Filter by minimum activation score
- View prompt and generated text contexts
- Explore which text patterns activate specific SAE features

## Setup

```bash
npm install
npm run dev
```

Dashboard runs at [http://localhost:3000](http://localhost:3000)

## Usage

### Router Mode

1. Generate router data:
   ```bash
   cd ..
   uv run export_expert_dashboard.py --layer 21 --samples 120000 --keep 12 --out expert_dashboard_layer21.json
   ```

2. Click "Router" tab in dashboard
3. Click "Load Router" and select `expert_dashboard_layer21.json`
4. Browse experts, adjust filters, search contexts

### Feature Mode

#### Single Feature
1. Generate feature data:
   ```bash
   cd ..
   uv run export_sae_features.py --feature 42 --samples 50000 --out feature_42.json
   ```

2. Click "Features" tab in dashboard
3. Click "Load Feature" and select `feature_42.json`
4. Adjust min score slider to filter activations
5. Color intensity shows activation strength on generated tokens

#### Multiple Features (Comparison)
1. Generate multiple features at once:
   ```bash
   cd ..
   uv run export_all_features.py --features 5,10,15,20,25,30 --samples 50000 --out-dir feature_exports
   ```

2. Click "Features" tab in dashboard
3. Click "Load Feature" and select **multiple** JSON files (Ctrl/Cmd+Click)
4. Use dropdown to switch between loaded features
5. Compare activation patterns across different features

## Data Format

### Router JSON
```json
{
  "meta": {"layer": 21, "n_experts": 32, "keep": 12, "samples": 120000},
  "experts": [
    {
      "id": 0,
      "stats": {"count": 4247, "meanTopP": 0.31, "meanEntropy": 1.37},
      "top": [{"p": 0.427, "prompt": "...", "generated": "..."}]
    }
  ]
}
```

### Feature JSON
```json
{
  "meta": {"feature_id": 42, "d_sae": 256, "samples": 5000, "k_keep": 8},
  "activations": [
    {"prompt": "...", "generated": "...", "score": 15.24, "layer": 21, "token_idx": 646}
  ]
}
```

## Tech Stack

- Next.js 15 with App Router
- React 19
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Recharts for visualizations
- Lucide icons
