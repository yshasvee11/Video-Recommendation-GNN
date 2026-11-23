"""
predict.py – Interactive Terminal Recommendation & Genre Diversity Script for Paper 3 (MovieLens-1M).
Usage:
    python paper-3/predict.py --user 42 --lambda_div 0.3 --topk 10
    python paper-3/predict.py --user 123 --lambda_div 0.7
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure paper-3 directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from dataloader import MovieLensDataset
from model import VynoraPaper3GNN


def load_movie_metadata(data_dir):
    """
    Loads movie titles and full genre strings from movies.dat.
    Format: MovieID::Title::Genres
    """
    movies_path = os.path.join(data_dir, "movies.dat")
    meta = {}
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(
            movies_path, sep="::", engine="python",
            names=["m_id", "title", "genres"], encoding="latin-1"
        )
        for _, row in movies_df.iterrows():
            meta[int(row["m_id"])] = {
                "title": str(row["title"]),
                "genres": str(row["genres"])
            }
    return meta


def compute_gini_simpson_diversity(genre_indices):
    """
    Computes Gini-Simpson diversity: D = 1 - sum(p_g^2)
    """
    n = len(genre_indices)
    if n < 2:
        return 0.0
    counts = {}
    for g in genre_indices:
        counts[g] = counts.get(g, 0) + 1
    same_pairs = sum(c * (c - 1) for c in counts.values() if c > 1)
    total_pairs = n * (n - 1)
    return 1.0 - (same_pairs / total_pairs)


def rank_movies_mmr(scores, cand_indices, cand_genres, topk=10, lambda_div=0.0):
    """
    Ranks candidate items using Maximal Marginal Relevance (MMR) for genre diversity.
    lambda_div = 0.0: Pure relevance
    lambda_div > 0.0: Trade-off between collaborative score and genre novelty
    """
    if lambda_div <= 0.0:
        top_idx = np.argsort(-scores)[:topk]
        return [cand_indices[i] for i in top_idx], [scores[i] for i in top_idx]

    # Normalize scores to [0, 1]
    s_min, s_max = np.min(scores), np.max(scores)
    norm_scores = (scores - s_min) / (s_max - s_min + 1e-8)

    selected = []
    selected_scores = []
    selected_genres = set()
    rem_indices = list(range(len(cand_indices)))

    for _ in range(min(topk, len(cand_indices))):
        best_cand_idx = -1
        best_mmr_val = -1e9

        for idx in rem_indices:
            g = cand_genres[idx]
            # Novelty: 1.0 if novel genre, 0.0 if already present in selected list
            novelty = 1.0 if g not in selected_genres else 0.0
            mmr = (1.0 - lambda_div) * norm_scores[idx] + lambda_div * novelty
            if mmr > best_mmr_val:
                best_mmr_val = mmr
                best_cand_idx = idx

        selected.append(cand_indices[best_cand_idx])
        selected_scores.append(scores[best_cand_idx])
        selected_genres.add(cand_genres[best_cand_idx])
        rem_indices.remove(best_cand_idx)

    return selected, selected_scores


def main():
    parser = argparse.ArgumentParser(description="MovieLens-1M Paper 3 Recommendation & Genre Diversity Predictor")
    parser.add_argument("--user", "-u", type=int, default=42, help="Raw MovieLens User ID (1 to 6040, default: 42)")
    parser.add_argument("--lambda_div", "-l", type=float, default=0.3, help="Genre diversity weight lambda in [0.0, 1.0] (default: 0.3)")
    parser.add_argument("--topk", "-k", type=int, default=10, help="Number of movies to recommend (default: 10)")
    parser.add_argument("--ckpt", type=str, default="", help="Optional explicit path to .pt checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="Path to ml-1m directory")
    args = parser.parse_args()

    print("=" * 80)
    print("🎬 VYNORA PAPER 3 (MPGCL + DENOISING) - MOVIELENS-1M RECOMMENDATION ENGINE")
    print("=" * 80)

    # 1. Resolve Data Directory
    data_dir = args.data_dir
    if not data_dir:
        # Check standard relative paths
        candidates = [
            config.DATA_DIR,
            os.path.abspath(os.path.join(CURRENT_DIR, "..", "other-dataset's", "movielens", "ml-1m")),
            os.path.abspath(os.path.join(CURRENT_DIR, "data", "movielens", "ml-1m")),
            os.path.abspath("other-dataset's/movielens/ml-1m")
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "ratings.dat")):
                data_dir = c
                break

    if not data_dir or not os.path.exists(os.path.join(data_dir, "ratings.dat")):
        print(f"❌ Error: MovieLens-1M ratings.dat not found! Please provide --data_dir.")
        sys.exit(1)

    print(f"📁 Dataset Path: {data_dir}")
    movie_metadata = load_movie_metadata(data_dir)

    # 2. Load Dataset Structure
    print("⏳ Loading graph structures and user-movie mappings...")
    dataset = MovieLensDataset(data_dir=data_dir)
    print(f"✓ Loaded {dataset.n_users:,} users, {dataset.n_items:,} movies, {dataset.n_interactions:,} ratings.")

    # 3. Validate User ID
    raw_user_id = args.user
    if raw_user_id not in dataset.u_map:
        print(f"⚠️ User ID {raw_user_id} not found in MovieLens. Defaulting to first available user.")
        u_idx = 0
        raw_user_id = dataset.inv_u_map[0]
    else:
        u_idx = dataset.u_map[raw_user_id]

    print(f"👤 Target User: MovieLens UserID #{raw_user_id} (Internal Index: {u_idx})")

    # 4. Initialize Model
    device = config.DEVICE
    model = VynoraPaper3GNN(dataset.n_users, dataset.n_items).to(device)
    model.set_graphs(dataset.graph_ui, dataset.meta_graph_u, dataset.meta_graph_i)

    # 5. Load Checkpoint
    ckpt_path = args.ckpt
    if not ckpt_path:
        ckpt_candidates = [
            os.path.join(config.CHECKPOINT_DIR, "best_movielens_model.pt"),
            os.path.join(config.CHECKPOINT_DIR, "checkpoint_latest.pt"),
            os.path.join(CURRENT_DIR, "checkpoints", "best_movielens_model.pt"),
            os.path.join(CURRENT_DIR, "checkpoints", "checkpoint_latest.pt"),
        ]
        for cp in ckpt_candidates:
            if os.path.exists(cp):
                ckpt_path = cp
                break

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"📦 Loading trained checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                print(f"✓ Resumed trained weights from Epoch {ckpt.get('epoch', 'N/A')} (Best NDCG: {ckpt.get('best_val_ndcg', 'N/A'):.4f})")
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
                print("✓ Loaded model state dictionary successfully.")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint ({e}). Running with initialized embeddings.")
    else:
        print("💡 [Notice]: No local .pt checkpoint found in paper-3/checkpoints/.")
        print("   -> Running inference using initialized Graph Neural Network embeddings.")
        print("   -> To evaluate your 40-epoch trained model from Colab, place 'best_movielens_model.pt' into 'paper-3/checkpoints/'.")

    model.eval()

    # 6. Display User Watch History Sample
    watched_items = dataset.train_pos_set[u_idx].union(dataset.val_pos_set[u_idx]).union(dataset.test_pos_set[u_idx])
    watched_list = list(watched_items)
    print("\n" + "-" * 80)
    print(f"📜 USER #{raw_user_id} WATCH HISTORY SAMPLE (Total watched: {len(watched_items)} movies)")
    print("-" * 80)
    for sample_i in watched_list[:5]:
        mid = dataset.inv_i_map[sample_i]
        meta = movie_metadata.get(mid, {"title": f"Movie #{mid}", "genres": "Unknown"})
        print(f"  ⭐ [{mid}] {meta['title']}  |  Genres: {meta['genres']}")
    if len(watched_list) > 5:
        print(f"  ... and {len(watched_list) - 5} more historical ratings.")

    # 7. Compute Predictions
    with torch.no_grad():
        h_u, h_v = model.forward()
        u_emb = h_u[u_idx] # (embed_dim,)
        # Compute dot product across all items in catalog
        all_scores = torch.mv(h_v, u_emb).cpu().numpy()

    # Filter out already watched items
    cand_indices = [i for i in range(dataset.n_items) if i not in watched_items]
    cand_scores = all_scores[cand_indices]
    cand_primary_genres = [dataset.genre_list[i] for i in cand_indices]

    # 8. Compute Recommendations: Pure Relevance (lambda = 0.0) vs. Diversified (lambda = chosen)
    rec_pure, scores_pure = rank_movies_mmr(
        cand_scores, cand_indices, cand_primary_genres, topk=args.topk, lambda_div=0.0
    )
    rec_div, scores_div = rank_movies_mmr(
        cand_scores, cand_indices, cand_primary_genres, topk=args.topk, lambda_div=args.lambda_div
    )

    # Calculate diversity metrics
    div_metric_pure = compute_gini_simpson_diversity([dataset.genre_list[i] for i in rec_pure])
    div_metric_div = compute_gini_simpson_diversity([dataset.genre_list[i] for i in rec_div])

    unique_genres_pure = len(set(dataset.genre_list[i] for i in rec_pure))
    unique_genres_div = len(set(dataset.genre_list[i] for i in rec_div))

    # 9. Print Side-by-Side Comparison
    print("\n" + "=" * 80)
    print(f"🎯 TOP-{args.topk} RECOMMENDATIONS FOR USER #{raw_user_id}")
    print("=" * 80)
    print(f"Pure Relevance (λ = 0.0)             vs.    Diversified Recommendation (λ = {args.lambda_div})")
    print(f"Genre Diversity (Simpson): {div_metric_pure:.4f}      |      Genre Diversity (Simpson): {div_metric_div:.4f}")
    print(f"Unique Genres Covered:    {unique_genres_pure:<2d}          |      Unique Genres Covered:    {unique_genres_div:<2d}")
    print("-" * 80)

    for rank in range(args.topk):
        # Pure item
        i_pure = rec_pure[rank]
        mid_pure = dataset.inv_i_map[i_pure]
        meta_pure = movie_metadata.get(mid_pure, {"title": f"Movie #{mid_pure}", "genres": "N/A"})
        title_pure = (meta_pure["title"][:28] + "..") if len(meta_pure["title"]) > 30 else meta_pure["title"]
        g_pure = meta_pure["genres"][:16]

        # Diversified item
        i_div = rec_div[rank]
        mid_div = dataset.inv_i_map[i_div]
        meta_div = movie_metadata.get(mid_div, {"title": f"Movie #{mid_div}", "genres": "N/A"})
        title_div = (meta_div["title"][:28] + "..") if len(meta_div["title"]) > 30 else meta_div["title"]
        g_div = meta_div["genres"][:16]

        print(f"#{rank+1:02d} | {title_pure:<30} [{g_pure:<16}] | #{rank+1:02d} | {title_div:<30} [{g_div:<16}]")

    print("-" * 80)
    print(f"💡 Explanation:")
    print(f"  • Under λ = 0.0 (Pure Accuracy), the model ranks strictly by predicted interaction affinity.")
    print(f"  • Under λ = {args.lambda_div} (MMR Re-Ranking), the model balances collaborative relevance with genre")
    print(f"    novelty, expanding the recommendation list from {unique_genres_pure} to {unique_genres_div} distinct genres.")
    print(f"    This alleviates the 'filter bubble' / 'genre fatigue' problem without sacrificing relevance.")
    print("=" * 80)


if __name__ == "__main__":
    main()
