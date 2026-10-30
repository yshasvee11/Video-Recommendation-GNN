"""
diversity_benchmark.py  –  Vynora 2.0
Benchmark the Accuracy vs. Diversity Trade-off (Paper 5 Implementation).
Evaluates multiple lambda_div values across test/validation users to chart
Recall@K, NDCG@K, Publisher Diversity@K, Content Diversity@K, and Catalog Coverage@K.

Reference: Paper 5 (Chun et al., Expert Systems with Applications 2024)

Usage:
    python diversity_benchmark.py --dataset takatak --mode val
    python diversity_benchmark.py --dataset wechat --mode test
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import config
from dataloader import VynoraDataset
from model import VynoraGNN
from predict import load_model_and_dataset
from evaluate import evaluate, format_results


def run_benchmark(dataset_name: str = config.DATASET,
                  mode: str = "val",
                  lambdas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
                  diversity_type: str = "publisher",
                  diversity_method: str = "mmr",
                  topk_list=(10, 20)):
    print(f"\n{'='*70}")
    print(f"  Vynora 2.0: Paper 5 Diversity Benchmark")
    print(f"  Dataset: {dataset_name} | Mode: {mode} | Method: {diversity_method} | Type: {diversity_type}")
    print(f"{'='*70}\n")

    model, dataset, graph_uv, graph_ua = load_model_and_dataset(dataset_name)

    results_table = []
    k_primary = topk_list[0]

    for lam in lambdas:
        t0 = time.time()
        print(f"Evaluating \u03bb = {lam:.2f} ...", end="", flush=True)

        res = evaluate(
            model=model,
            dataset=dataset,
            graph_uv=graph_uv,
            graph_ua=graph_ua,
            mode=mode,
            topk_list=list(topk_list),
            diversity_lambda=lam,
            diversity_type=diversity_type,
            diversity_method=diversity_method,
            compute_diversity=True
        )
        elapsed = time.time() - t0
        print(f" ({elapsed:.1f}s)")

        row = {
            "lambda": float(lam),
            "Recall@10": res["Recall"].get(10, 0.0),
            "NDCG@10": res["NDCG"].get(10, 0.0),
            "Precision@10": res["Precision"].get(10, 0.0),
            "MRR@10": res["MRR"].get(10, 0.0),
            "Publisher_Diversity@10": res.get("Publisher_Diversity", {}).get(10, 0.0),
            "Content_Diversity@10": res.get("Content_Diversity", {}).get(10, 0.0),
            "Catalog_Coverage@10": res.get("Catalog_Coverage", {}).get(10, 0.0),
        }
        if 20 in topk_list:
            row["Recall@20"] = res["Recall"].get(20, 0.0)
            row["NDCG@20"] = res["NDCG"].get(20, 0.0)
            row["Publisher_Diversity@20"] = res.get("Publisher_Diversity", {}).get(20, 0.0)

        results_table.append(row)

    # Print summary table
    print(f"\n{'='*82}")
    print(f"{'Lambda':>8} | {'Recall@10':>10} | {'NDCG@10':>10} | {'PubDiv@10':>10} | {'ConDiv@10':>10} | {'Coverage@10':>11}")
    print(f"{'-'*82}")
    for row in results_table:
        print(f"{row['lambda']:>8.2f} | "
              f"{row['Recall@10']:>10.4f} | "
              f"{row['NDCG@10']:>10.4f} | "
              f"{row['Publisher_Diversity@10']:>10.4f} | "
              f"{row['Content_Diversity@10']:>10.4f} | "
              f"{row['Catalog_Coverage@10']:>11.4f}")
    print(f"{'='*82}\n")

    # Save to JSON
    out_file = os.path.join(config.RESULT_DIR, f"diversity_tradeoff_{dataset_name}_{diversity_method}.json")
    with open(out_file, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"[Done] Saved benchmark results to: {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Vynora 2.0 – Paper 5 Diversity Benchmark")
    parser.add_argument("--dataset", type=str, default=config.DATASET, help="takatak | wechat")
    parser.add_argument("--mode", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--method", type=str, default="mmr", choices=["mmr", "score_rerank"])
    parser.add_argument("--type", type=str, default="publisher", choices=["publisher", "content", "hybrid"])
    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset,
        mode=args.mode,
        diversity_method=args.method,
        diversity_type=args.type
    )


if __name__ == "__main__":
    main()

