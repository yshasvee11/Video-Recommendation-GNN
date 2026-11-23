"""
recommend.py - Unified Interactive Terminal Recommendation CLI
Allows passing --user and --lambda_div to generate personalized recommendations
with a side-by-side comparison between pure relevance (lambda=0.0) and diversified recommendations.

Usage:
    python recommend.py --user 42 --lambda_div 0.3 --dataset takatak
    python recommend.py --user 5 --lambda_div 0.5 --dataset wechat
    python recommend.py --user 10 --lambda_div 0.0  # pure relevance baseline
"""

import os
import sys
import argparse

# Add project/vynora2 to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VYNORA2_DIR = os.path.join(BASE_DIR, "project", "vynora2")
if VYNORA2_DIR not in sys.path:
    sys.path.insert(0, VYNORA2_DIR)

import config
import torch
import numpy as np
from dataloader import VynoraDataset
from model import VynoraGNN
from predict import load_model_and_dataset, recommend_for_user


def main():
    parser = argparse.ArgumentParser(description="Vynora Interactive Recommendation CLI with Lambda Diversity")
    parser.add_argument("--user", type=int, default=42, help="User ID to generate recommendations for (e.g. 42, 5, 0)")
    parser.add_argument("--lambda_div", "--lambda", "-l", type=float, default=0.3, dest="lambda_div",
                        help="Diversity trade-off parameter lambda in [0.0, 1.0]. (0.0 = Pure Relevance, 1.0 = Max Diversity)")
    parser.add_argument("--dataset", "-d", type=str, default="takatak", choices=["takatak", "wechat"],
                        help="Dataset to recommend from ('takatak' or 'wechat')")
    parser.add_argument("--topk", "-k", type=int, default=10, help="Number of items to recommend (default: 10)")
    parser.add_argument("--diversity_type", type=str, default="publisher", choices=["publisher", "content", "hybrid"],
                        help="Dimension of diversity: 'publisher' (creator variety) or 'content'")
    parser.add_argument("--diversity_method", type=str, default="mmr", choices=["mmr", "score_rerank"],
                        help="Re-ranking algorithm: 'mmr' (Maximal Marginal Relevance) or 'score_rerank'")

    args = parser.parse_args()

    print("=" * 70)
    print("  VYNORA INTERACTIVE HOME SCREEN RECOMMENDATION SYSTEM")
    print(f"  Target User   : {args.user}")
    print(f"  Dataset       : {args.dataset.upper()}")
    print(f"  Diversity λ   : {args.lambda_div:.2f} ({'Pure Relevance' if args.lambda_div == 0.0 else 'Diversified Mode'})")
    print(f"  Top-K Items   : {args.topk}")
    print("=" * 70)

    # 1. Load Model and Graph
    model, dataset, graph_uv, graph_ua = load_model_and_dataset(args.dataset, allow_untrained=True)

    if args.user < 0 or args.user >= dataset.n_users:
        print(f"Error: User ID {args.user} out of bounds! Valid range is [0, {dataset.n_users - 1}].")
        return

    # 2. Generate Recommendations
    rec_data = recommend_for_user(
        user_id=args.user,
        model=model,
        dataset=dataset,
        graph_uv=graph_uv,
        graph_ua=graph_ua,
        topk=args.topk,
        exclude_seen=True,
        diversity_lambda=args.lambda_div,
        diversity_type=args.diversity_type,
        diversity_method=args.diversity_method
    )

    recs = rec_data["results"]
    div_m = rec_data["diversity_metrics"]
    base_recs = rec_data["baseline_results"]
    base_div_m = rec_data["baseline_diversity_metrics"]

    # 3. Print Results Table
    if args.lambda_div > 0.0:
        print(f"\n{'Rank':>4} | {'Pure Relevance (λ=0.0)':^26} | {'Diversified (λ=' + str(args.lambda_div) + ')':^26}")
        print(f"{'':>4} | {'Video ID':>9} {'Vlogger':>8} {'Score':>7} | {'Video ID':>9} {'Vlogger':>8} {'Score':>7}")
        print("-" * 62)
        for r in range(args.topk):
            b_vid = base_recs[r]['video_id'] if r < len(base_recs) else '-'
            b_vlog = base_recs[r]['vlogger_id'] if r < len(base_recs) else '-'
            b_sc = base_recs[r]['score'] if r < len(base_recs) else 0.0

            d_vid = recs[r]['video_id'] if r < len(recs) else '-'
            d_vlog = recs[r]['vlogger_id'] if r < len(recs) else '-'
            d_sc = recs[r]['score'] if r < len(recs) else 0.0

            print(f"{r+1:>4} | {b_vid:>9} {b_vlog:>8} {b_sc:>7.3f} | {d_vid:>9} {d_vlog:>8} {d_sc:>7.3f}")
        print("-" * 62)
    else:
        print(f"\n{'Rank':>4} | {'Video ID':>10} | {'Creator/Vlogger ID':>20} | {'Predicted Score':>16}")
        print("-" * 58)
        for rec in recs:
            print(f"{rec['rank']:>4} | {rec['video_id']:>10} | {rec['vlogger_id']:>20} | {rec['score']:>16.4f}")
        print("-" * 58)

    # 4. Print Diversity Metrics Comparison
    print(f"\n📊 Diversity Metrics Breakdown:")
    print(f"  • Publisher Diversity (Gini-Simpson Index): {div_m['publisher_diversity']:.4f} (1.0 = All distinct creators)")
    print(f"  • Unique Content Creators in Top-{args.topk}    : {div_m['unique_vloggers']} / {args.topk}")
    print(f"  • Creator Shannon Entropy                 : {div_m['vlogger_entropy']:.4f} bits")

    if args.lambda_div > 0.0:
        pub_diff = (div_m['publisher_diversity'] - base_div_m['publisher_diversity']) / (base_div_m['publisher_diversity'] + 1e-8) * 100.0
        vlog_diff = div_m['unique_vloggers'] - base_div_m['unique_vloggers']
        print(f"\n📈 Impact of Setting λ={args.lambda_div} vs Pure Relevance Baseline (λ=0.0):")
        print(f"  • Publisher Diversity Shift : {pub_diff:+.1f}% ({base_div_m['publisher_diversity']:.4f} -> {div_m['publisher_diversity']:.4f})")
        print(f"  • Creator Variety Change    : {vlog_diff:+d} unique creators ({base_div_m['unique_vloggers']} -> {div_m['unique_vloggers']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
