"""
predict.py – Clean & Minimal Recommendation Predictor for Paper 3 (MovieLens-1M).
Usage:
    python paper-3/predict.py --user 42 --lambda_div 0.3
    python paper-3/predict.py --user 123 --lambda_div 0.0
"""

import os
import sys
import io
import contextlib
import argparse
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from dataloader import MovieLensDataset
from model import VynoraPaper3GNN


def load_movie_metadata(data_dir):
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


def compute_diversity(genre_indices):
    n = len(genre_indices)
    if n < 2:
        return 0.0
    counts = {}
    for g in genre_indices:
        counts[g] = counts.get(g, 0) + 1
    same_pairs = sum(c * (c - 1) for c in counts.values() if c > 1)
    total_pairs = n * (n - 1)
    return 1.0 - (same_pairs / total_pairs)


def rank_mmr(scores, cand_indices, cand_genres, topk=10, lambda_div=0.0):
    if lambda_div <= 0.0:
        top_idx = np.argsort(-scores)[:topk]
        return [cand_indices[i] for i in top_idx], [scores[i] for i in top_idx]

    s_min, s_max = np.min(scores), np.max(scores)
    norm_scores = (scores - s_min) / (s_max - s_min + 1e-8)

    selected = []
    selected_scores = []
    selected_genres = set()
    rem = list(range(len(cand_indices)))

    for _ in range(min(topk, len(cand_indices))):
        best_idx = -1
        best_val = -1e9
        for i in rem:
            g = cand_genres[i]
            novelty = 1.0 if g not in selected_genres else 0.0
            val = (1.0 - lambda_div) * norm_scores[i] + lambda_div * novelty
            if val > best_val:
                best_val = val
                best_idx = i

        selected.append(cand_indices[best_idx])
        selected_scores.append(scores[best_idx])
        selected_genres.add(cand_genres[best_idx])
        rem.remove(best_idx)

    return selected, selected_scores


def main():
    parser = argparse.ArgumentParser(description="MovieLens-1M Paper 3 Recommendation Predictor")
    parser.add_argument("--user", "-u", type=int, default=42, help="MovieLens User ID (1-6040)")
    parser.add_argument("--lambda_div", "-l", type=float, default=0.3, help="Genre diversity parameter [0.0 - 1.0]")
    parser.add_argument("--topk", "-k", type=int, default=10, help="Number of items to recommend")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="Path to ml-1m directory")
    parser.add_argument("--show-history", action="store_true", help="Show user's watch history sample")
    args = parser.parse_args()

    # Resolve Data Directory
    data_dir = args.data_dir
    if not data_dir:
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
        print(f"Error: MovieLens dataset not found. Provide --data_dir.")
        sys.exit(1)

    movie_metadata = load_movie_metadata(data_dir)

    # Load dataset quietly
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = MovieLensDataset(data_dir=data_dir)

    raw_user_id = args.user
    if raw_user_id not in dataset.u_map:
        print(f"Error: User ID #{raw_user_id} not found in MovieLens (valid range: 1 to {dataset.n_users}).")
        sys.exit(1)

    u_idx = dataset.u_map[raw_user_id]

    # Initialize model
    model = VynoraPaper3GNN(dataset.n_users, dataset.n_items).to(config.DEVICE)
    model.set_graphs(dataset.graph_ui, dataset.meta_graph_u, dataset.meta_graph_i)

    # Load checkpoint quietly if exists
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
        try:
            ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
        except Exception:
            pass

    model.eval()

    # User History (Optional)
    watched_items = dataset.train_pos_set[u_idx].union(dataset.val_pos_set[u_idx]).union(dataset.test_pos_set[u_idx])
    if args.show_history:
        print(f"\nUser #{raw_user_id} History ({len(watched_items)} total watched):")
        for sample_i in list(watched_items)[:5]:
            mid = dataset.inv_i_map[sample_i]
            meta = movie_metadata.get(mid, {"title": f"Movie #{mid}", "genres": "Unknown"})
            print(f"  • {meta['title']} ({meta['genres']})")

    # Compute Predictions
    with torch.no_grad():
        h_u, h_v = model.forward()
        scores = torch.mv(h_v, h_u[u_idx]).cpu().numpy()

    cand_indices = [i for i in range(dataset.n_items) if i not in watched_items]
    cand_scores = scores[cand_indices]
    cand_genres = [dataset.genre_list[i] for i in cand_indices]

    # Mode 1: Pure Accuracy Only (lambda == 0)
    if args.lambda_div <= 0.0:
        rec, _ = rank_mmr(cand_scores, cand_indices, cand_genres, topk=args.topk, lambda_div=0.0)
        div = compute_diversity([dataset.genre_list[i] for i in rec])
        u_genres = len(set(dataset.genre_list[i] for i in rec))

        print(f"\nRecommendations for User #{raw_user_id} (λ = 0.0)")
        print(f"{'#':<3} {'Title':<36} {'Genre':<22}")
        print("-" * 64)
        for rank, item_idx in enumerate(rec):
            mid = dataset.inv_i_map[item_idx]
            meta = movie_metadata.get(mid, {"title": f"Movie #{mid}", "genres": "N/A"})
            title = (meta["title"][:33] + "..") if len(meta["title"]) > 35 else meta["title"]
            genre = (meta["genres"][:20] + "..") if len(meta["genres"]) > 22 else meta["genres"]
            print(f"{rank+1:02d}  {title:<36} {genre:<22}")
        print("-" * 64)
        print(f"Simpson Diversity: {div:.2f} ({u_genres} unique genres)\n")
        return

    # Mode 2: Comparative Display (Relevance vs. Diversified)
    rec_pure, _ = rank_mmr(cand_scores, cand_indices, cand_genres, topk=args.topk, lambda_div=0.0)
    rec_div, _ = rank_mmr(cand_scores, cand_indices, cand_genres, topk=args.topk, lambda_div=args.lambda_div)

    div_pure = compute_diversity([dataset.genre_list[i] for i in rec_pure])
    div_div = compute_diversity([dataset.genre_list[i] for i in rec_div])

    u_genres_pure = len(set(dataset.genre_list[i] for i in rec_pure))
    u_genres_div = len(set(dataset.genre_list[i] for i in rec_div))

    print(f"\nRecommendations for User #{raw_user_id} (λ = {args.lambda_div})")
    print(f"{'#':<3} {'Relevance (λ = 0.0)':<32} {'Genre':<16} |  {'Diversified (λ = ' + str(args.lambda_div) + ')':<32} {'Genre':<16}")
    print("-" * 105)

    for rank in range(args.topk):
        # Pure
        mid_p = dataset.inv_i_map[rec_pure[rank]]
        meta_p = movie_metadata.get(mid_p, {"title": f"Movie #{mid_p}", "genres": "N/A"})
        t_p = (meta_p["title"][:29] + "..") if len(meta_p["title"]) > 31 else meta_p["title"]
        g_p = (meta_p["genres"][:14] + "..") if len(meta_p["genres"]) > 16 else meta_p["genres"]

        # Div
        mid_d = dataset.inv_i_map[rec_div[rank]]
        meta_d = movie_metadata.get(mid_d, {"title": f"Movie #{mid_d}", "genres": "N/A"})
        t_d = (meta_d["title"][:29] + "..") if len(meta_d["title"]) > 31 else meta_d["title"]
        g_d = (meta_d["genres"][:14] + "..") if len(meta_d["genres"]) > 16 else meta_d["genres"]

        print(f"{rank+1:02d}  {t_p:<32} {g_p:<16} |  {t_d:<32} {g_d:<16}")

    print("-" * 105)
    print(f"     Diversity: {div_pure:.2f} ({u_genres_pure} genres)                |  Diversity: {div_div:.2f} ({u_genres_div} genres)\n")


if __name__ == "__main__":
    main()
