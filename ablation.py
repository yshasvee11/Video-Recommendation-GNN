"""
ablation.py  –  Vynora 2.0
Eight-variant publication ablation matrix (M1 through M8) across research components:
  M1. LightGCN Baseline              (pure user-video collaborative graph, no creator view)
  M2. VA-GNN (Fixed Gate λ=0.5)      (dual view with fixed 50/50 blend, creator BPR)
  M3. VA-GNN (Bilinear Adaptive Gate)(VA-GNN Eq 17: w = sigmoid(e_v^T Q e_{p_v}))
  M4. VA-GNN Core (+ 3-Entity CL)    (Bilinear Gate + User/Video/Creator InfoNCE)
  M5. VA-GNN + Sagittarius WBPR      (Log-frequency interaction weighting: w_{uv} = log(1 + c_{uv}))
  M6. VA-GNN + MPGCL Meta-Paths      (Sparse UVU & VUV homogeneous graph contrastive regularization)
  M7. VA-GNN + Soft Filtering        (End-to-end differentiable relevance attention)
  M8. Full Vynora + MMR Diversity    (Core backbone + MMR publisher diversification at λ=0.3)
"""

import json
import os
import config

VARIANTS = {
    "M1_LightGCN":              {"CL_REG": 0.0, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.0, "USE_ADAPTIVE_GATE": False, "LAMBDA_VLOGGER": 1.0, "USE_FREQUENCY_WEIGHTING": False, "USE_MPGCL": False},
    "M2_VAGNN_FixedGate":       {"CL_REG": 0.0, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.1, "USE_ADAPTIVE_GATE": False, "LAMBDA_VLOGGER": 0.5, "USE_FREQUENCY_WEIGHTING": False, "USE_MPGCL": False},
    "M3_VAGNN_BilinearGate":    {"CL_REG": 0.0, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.1, "USE_ADAPTIVE_GATE": True,  "USE_FREQUENCY_WEIGHTING": False, "USE_MPGCL": False},
    "M4_VAGNN_Core_3EntityCL":  {"CL_REG": 0.1, "USE_3ENTITY_CL": True, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.1, "USE_ADAPTIVE_GATE": True, "USE_FREQUENCY_WEIGHTING": False, "USE_MPGCL": False},
    "M5_VAGNN_Sagittarius_WBPR":{"CL_REG": 0.1, "USE_3ENTITY_CL": True, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.1, "USE_ADAPTIVE_GATE": True, "USE_FREQUENCY_WEIGHTING": True, "USE_MPGCL": False},
    "M6_VAGNN_MPGCL":           {"CL_REG": 0.1, "USE_3ENTITY_CL": True, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.1, "USE_ADAPTIVE_GATE": True, "USE_FREQUENCY_WEIGHTING": True, "USE_MPGCL": True},
    "M7_VAGNN_SoftFilter":      {"CL_REG": 0.1, "USE_3ENTITY_CL": True, "DENOISE_REG": 0.05,"VLOGGER_REG": 0.1, "USE_ADAPTIVE_GATE": True, "USE_FREQUENCY_WEIGHTING": True, "USE_SOFT_FILTER": True},
    "M8_Full_Vynora_MMR":       {"CL_REG": 0.1, "USE_3ENTITY_CL": True, "DENOISE_REG": 0.05,"VLOGGER_REG": 0.1, "USE_ADAPTIVE_GATE": True, "USE_FREQUENCY_WEIGHTING": True, "DIVERSITY_LAMBDA": 0.3},
}


def run_ablation():
    all_results = {}

    for variant_name, overrides in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f" Ablation Variant: {variant_name}")
        print(f" Overrides: {overrides}")
        print(f"{'='*70}")

        # Apply overrides to config
        for k, v in overrides.items():
            setattr(config, k, v)

        # Override result path per variant
        original_result_dir = config.RESULT_DIR
        config.RESULT_DIR = os.path.join(original_result_dir, "ablation")
        os.makedirs(config.RESULT_DIR, exist_ok=True)

        # Override checkpoint path per variant
        original_ckpt_dir = config.CKPT_DIR
        config.CKPT_DIR = os.path.join(original_ckpt_dir, variant_name)
        os.makedirs(config.CKPT_DIR, exist_ok=True)

        # Import train fresh each time (avoid stale state)
        import importlib
        import train as train_module
        importlib.reload(train_module)

        results = train_module.train()
        all_results[variant_name] = results

        # Restore
        config.RESULT_DIR = original_result_dir
        config.CKPT_DIR   = original_ckpt_dir

    # Print summary table
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    header = f"{'Variant':<25}" + "".join(
        f"R@{k:<6}NDCG@{k:<6}" for k in config.TOPK_LIST
    )
    print(header)
    print("-" * len(header))
    for variant_name, res in all_results.items():
        row = f"{variant_name:<25}"
        for k in config.TOPK_LIST:
            r = res.get("Recall", {}).get(k, 0.0)
            n = res.get("NDCG",   {}).get(k, 0.0)
            row += f"{r:.4f}  {n:.4f}  "
        print(row)

    # Save summary
    summary_path = os.path.join(config.RESULT_DIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            name: {m: {str(k): v for k, v in vals.items()}
                   for m, vals in res.items()}
            for name, res in all_results.items()
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_ablation()
