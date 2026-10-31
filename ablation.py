"""
ablation.py  –  Vynora 2.0
Six-variant ablation study of the Vynora RECOMMENDER model components.

SCOPE CLARIFICATION:
  This ablation studies the MAIN RECOMMENDER (VynoraGNN in model.py).
  It is NOT an ablation of the Paper 5 DiversificationGNN predictor or
  the diversity re-ranking module.

  The six variants toggle: vlogger-view, cross-view CL, and edge pruning.

Known result (from stored ablation_summary.json):
  - Variant A (Baseline):         Recall@10 = 0.0484
  - Variant B (+ Vlogger):        Recall@10 = 0.0471
  - Variant F (Full model):       Recall@10 = 0.0468
  The Full model underperforms the Baseline at current hyperparameters.
  This result is reported transparently in results/ablation_summary.json.

Variants:
  A. Baseline           (only L_rank, no denoising, single view)
  B. Baseline + vlogger (dual view, lambda scoring, L_vlogger)
  C. Baseline + CL      (dual view + cross-view contrastive, no denoising)
  D. Baseline + denoise (single view + edge pruning, no CL)
  E. B + D              (dual view + denoising, no CL)
  F. Full model         (dual view + CL + denoising)
"""

import json
import os
import config

# We monkey-patch config before importing train to control ablation flags.
# This is a clean approach when all hyper-params flow through config.

VARIANTS = {
    "A_baseline":          {"CL_REG": 0.0, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.0, "LAMBDA_VLOGGER": 1.0},
    "B_vlogger":           {"CL_REG": 0.0, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.1,  "LAMBDA_VLOGGER": 0.5},
    "C_contrastive":       {"CL_REG": 0.1, "DENOISE_REG": 0.0, "VLOGGER_REG": 0.1,  "LAMBDA_VLOGGER": 0.5},
    "D_denoising":         {"CL_REG": 0.0, "DENOISE_REG": 0.05,"VLOGGER_REG": 0.0,  "LAMBDA_VLOGGER": 1.0},
    "E_vlogger_denoise":   {"CL_REG": 0.0, "DENOISE_REG": 0.05,"VLOGGER_REG": 0.1,  "LAMBDA_VLOGGER": 0.5},
    "F_full":              {"CL_REG": 0.1, "DENOISE_REG": 0.05,"VLOGGER_REG": 0.1,  "LAMBDA_VLOGGER": 0.5},
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
