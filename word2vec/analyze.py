#!/usr/bin/env python3
"""
Analyze a trained Word2Vec model
================================

Run this after train.py to inspect what your model learned.

Reads the training output and prints:
  1. Weight snapshot statistics at key training steps
  2. Top similar word pairs (cosine similarity)
  3. Per-word convergence — which words settled first/last
  4. W vs W' comparison — how center and output embeddings differ
  5. Corpus and model statistics
  6. Model architecture summary

Usage:
    python analyze.py                        # reads from ./output/
    python analyze.py --data-dir ./my_run
    python analyze.py --save-json            # also writes JSON files
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SNAPSHOT_STEPS = [0, 200, 1000, 2000, 3000, 5000, 7500, 10000, 12000, 15000]

SEMANTIC_GROUPS = {
    "royalty": ["king", "queen", "prince", "princess", "crown", "throne",
                "palace", "kingdom", "knight"],
    "family": ["father", "mother", "son", "daughter", "child", "baby",
               "young", "old"],
    "people": ["man", "woman", "boy", "girl"],
    "animals": ["cat", "dog", "lion", "wolf", "bear", "rabbit", "horse",
                "bird", "fish", "animal"],
    "nature": ["sun", "moon", "star", "sky", "river", "ocean", "mountain",
               "forest", "tree", "flower", "garden", "field", "rain",
               "water", "fruit", "apple"],
    "actions": ["runs", "walks", "eats", "drinks", "plays", "sings",
                "dances", "flies", "swims", "rides", "fights", "hunts",
                "climbs", "sits", "sees", "sleeps", "picks", "lives",
                "loves", "rules", "shines", "wears"],
    "adjectives": ["big", "small", "strong", "weak", "fast", "slow",
                   "brave", "wise", "kind", "gentle", "proud", "happy",
                   "sad", "beautiful", "bright", "dark", "warm", "cold",
                   "wild", "old"],
    "places": ["house", "castle", "city", "village"],
}


# ── math helpers ─────────────────────────────────────────────────

def cosine_sim_matrix(W):
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = W / norms
    return normed @ normed.T


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── data loading ─────────────────────────────────────────────────

def load_step(data_dir, step):
    path = Path(data_dir) / "steps" / f"{step}.json"
    with open(path) as f:
        return json.load(f)


def load_meta(data_dir):
    path = Path(data_dir) / "run_metadata.json"
    if not path.exists():
        print(f"Error: {path} not found. Run train.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ── analyses ─────────────────────────────────────────────────────

def analyze_weight_snapshots(data_dir, vocab, available_steps):
    steps = [s for s in SNAPSHOT_STEPS if s in available_steps]
    print("=" * 60)
    print("1. WEIGHT SNAPSHOTS")
    print("=" * 60)

    step_matrices = {}
    for step in steps:
        data = load_step(data_dir, step)
        W = np.array(data["W"])
        Wp = np.array(data["W_prime"])
        step_matrices[step] = (W, Wp)
        w_norm = np.linalg.norm(W, axis=1).mean()
        wp_norm = np.linalg.norm(Wp, axis=1).mean()
        print(f"  Step {step:>5}:  W {W.shape}  avg‖w‖={w_norm:.3f}  "
              f"W' {Wp.shape}  avg‖w'‖={wp_norm:.3f}")

    return steps, step_matrices


def analyze_similarity(W_final, vocab, w2i, word_groups):
    print("\n" + "=" * 60)
    print("2. TOP SIMILAR WORD PAIRS")
    print("=" * 60)

    sim = cosine_sim_matrix(W_final)
    V = len(vocab)

    pairs = []
    for i in range(V):
        for j in range(i + 1, V):
            pairs.append((vocab[i], vocab[j], sim[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 20 most similar pairs:")
    print(f"  {'Word A':<12} {'Word B':<12} {'Cosine':>8}  Group")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 8}  {'─' * 15}")
    for a, b, s in pairs[:20]:
        ga = word_groups.get(a, "")
        gb = word_groups.get(b, "")
        group = ga if ga == gb else f"{ga}/{gb}" if ga and gb else ga or gb
        print(f"  {a:<12} {b:<12} {s:>8.4f}  {group}")

    return sim


def analyze_convergence(steps, step_matrices, W_final, vocab, w2i, word_groups):
    print("\n" + "=" * 60)
    print("3. PER-WORD CONVERGENCE")
    print("=" * 60)

    convergence = {}
    for word in vocab:
        idx = w2i[word]
        final_vec = W_final[idx]
        sims = []
        for step in steps:
            sim = cosine_similarity(step_matrices[step][0][idx], final_vec)
            sims.append(sim)

        settled = None
        for i, s in enumerate(sims):
            if s > 0.9:
                settled = steps[i]
                break
        convergence[word] = {"sims": sims, "settled_at": settled}

    fast = [(w, d["settled_at"]) for w, d in convergence.items()
            if d["settled_at"] is not None]
    fast.sort(key=lambda x: x[1])
    slow = sorted(fast, key=lambda x: x[1], reverse=True)
    never = [w for w, d in convergence.items() if d["settled_at"] is None]

    print(f"\n  Fastest converging (reach >0.9 similarity to final):")
    for w, s in fast[:10]:
        g = word_groups.get(w, "")
        print(f"    {w:<12} settled at step {s:>5}  [{g}]")

    print(f"\n  Slowest converging:")
    for w, s in slow[:10]:
        g = word_groups.get(w, "")
        print(f"    {w:<12} settled at step {s:>5}  [{g}]")

    if never:
        print(f"\n  Never reached >0.9: {', '.join(never[:15])}")

    return convergence


def analyze_w_vs_wprime(W_final, data_dir, vocab, w2i, word_groups):
    print("\n" + "=" * 60)
    print("4. W vs W' COMPARISON")
    print("=" * 60)

    final_data = load_step(data_dir, max(SNAPSHOT_STEPS))
    Wp_final = np.array(final_data["W_prime"])

    comparisons = []
    for i, word in enumerate(vocab):
        sim = cosine_similarity(W_final[i], Wp_final[i])
        comparisons.append({
            "word": word, "cosine": sim,
            "w_norm": float(np.linalg.norm(W_final[i])),
            "wprime_norm": float(np.linalg.norm(Wp_final[i])),
        })

    avg = np.mean([c["cosine"] for c in comparisons])
    print(f"\n  Average W vs W' cosine similarity: {avg:.4f}")

    by_sim = sorted(comparisons, key=lambda x: x["cosine"], reverse=True)
    print(f"\n  Most aligned (W ≈ W'):")
    for c in by_sim[:8]:
        g = word_groups.get(c["word"], "")
        print(f"    {c['word']:<12} cos={c['cosine']:.4f}  "
              f"‖w‖={c['w_norm']:.3f}  ‖w'‖={c['wprime_norm']:.3f}  [{g}]")

    print(f"\n  Least aligned (W ≠ W'):")
    for c in by_sim[-8:]:
        g = word_groups.get(c["word"], "")
        print(f"    {c['word']:<12} cos={c['cosine']:.4f}  "
              f"‖w‖={c['w_norm']:.3f}  ‖w'‖={c['wprime_norm']:.3f}  [{g}]")

    return comparisons


def analyze_corpus_stats(meta, w2i, word_groups):
    print("\n" + "=" * 60)
    print("5. CORPUS & MODEL STATISTICS")
    print("=" * 60)

    vocab = meta["vocab"]
    freqs = meta["word_frequencies"]
    config = meta["config"]
    raw = meta.get("raw_sentences", [])
    V = len(vocab)
    d = meta["embed_dim"]

    window = config["window_size"]
    pair_count = 0
    for sent in raw:
        words = [w for w in sent.split() if w in w2i]
        for i in range(len(words)):
            lo = max(0, i - window)
            hi = min(len(words), i + window + 1)
            pair_count += (hi - lo - 1)

    total_tokens = sum(freqs.values())

    print(f"\n  Sentences          : {len(raw)}")
    print(f"  Vocabulary size    : {V}")
    print(f"  Embedding dim      : {d}")
    print(f"  Total tokens       : {total_tokens}")
    print(f"  Training pairs     : {pair_count}")
    print(f"  Total parameters   : {V * d * 2:,}  (W: {V}×{d} + W': {V}×{d})")
    print(f"  Training steps     : {config['num_steps']:,}")
    print(f"  Learning rate      : {config['learning_rate']}")
    print(f"  Window size        : {window}")

    sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 words by frequency:")
    for w, c in sorted_freqs[:15]:
        g = word_groups.get(w, "")
        bar = "█" * min(c, 40)
        print(f"    {w:<12} {c:>4}  {bar}  [{g}]")


def analyze_architecture(meta):
    print("\n" + "=" * 60)
    print("6. MODEL ARCHITECTURE")
    print("=" * 60)

    V = len(meta["vocab"])
    d = meta["embed_dim"]

    print(f"""
  SkipGram(
    W       = Embedding({V}, {d})     # center embeddings  ({V * d:,} params)
    W_prime = Linear({d}, {V}, no bias)  # output embeddings  ({V * d:,} params)
  )
  Total parameters: {V * d * 2:,}

  Forward pass:
    one-hot [{V}]  →  W lookup  →  v_c [{d}]  →  W' @ v_c  →  logits [{V}]  →  softmax  →  P(·|center)

  Loss:  L = -log P(context_word | center_word)   (cross-entropy)
  Optimizer: SGD (manual gradient step, lr={meta['config']['learning_rate']})

  Scale comparison:
    This model            : {V * d * 2:>15,} parameters
    Word2Vec (Google 2013): {'900,000,000':>15} parameters
    GPT-2 Small           : {'124,000,000':>15} parameters
    BERT Base             : {'110,000,000':>15} parameters""")


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyze a trained Word2Vec model.")
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parent / "output"),
                    help="Path to training output (default: ./output/)")
    ap.add_argument("--save-json", action="store_true",
                    help="Also write analysis results as JSON files")
    args = ap.parse_args()

    data_dir = args.data_dir
    meta = load_meta(data_dir)
    vocab = meta["vocab"]
    w2i = {w: i for i, w in enumerate(vocab)}

    with open(Path(data_dir) / "step_index.json") as f:
        available_steps = set(json.load(f))

    word_groups = {}
    for group, words in SEMANTIC_GROUPS.items():
        for w in words:
            if w in w2i:
                word_groups[w] = group

    final_data = load_step(data_dir, max(
        s for s in SNAPSHOT_STEPS if s in available_steps))
    W_final = np.array(final_data["W"])

    steps, step_matrices = analyze_weight_snapshots(
        data_dir, vocab, available_steps)
    sim_matrix = analyze_similarity(W_final, vocab, w2i, word_groups)
    convergence = analyze_convergence(
        steps, step_matrices, W_final, vocab, w2i, word_groups)
    comparisons = analyze_w_vs_wprime(
        W_final, data_dir, vocab, w2i, word_groups)
    analyze_corpus_stats(meta, w2i, word_groups)
    analyze_architecture(meta)

    if args.save_json:
        out_dir = Path(data_dir)
        print("\n\nWriting JSON files...")

        group_order = ["royalty", "family", "people", "animals", "nature",
                       "actions", "adjectives", "places"]
        sorted_vocab = []
        for g in group_order:
            sorted_vocab.extend(
                sorted(w for w in vocab if word_groups.get(w) == g))
        sorted_vocab.extend(sorted(w for w in vocab if w not in word_groups))
        sorted_indices = [w2i[w] for w in sorted_vocab]
        sim_sorted = sim_matrix[np.ix_(sorted_indices, sorted_indices)]

        outputs = {
            "similarity_matrix.json": {
                "vocab_sorted": sorted_vocab,
                "groups": {w: word_groups.get(w, "other")
                           for w in sorted_vocab},
                "matrix": np.round(sim_sorted, 4).tolist(),
            },
            "word_convergence.json": {
                "steps": steps,
                "words": {
                    w: {
                        "similarities": [round(s, 4)
                                         for s in d["sims"]],
                        "group": word_groups.get(w, "other"),
                        "settled_at": d["settled_at"],
                    }
                    for w, d in convergence.items()
                },
            },
            "w_vs_wprime.json": {
                "vocab": vocab,
                "comparisons": [
                    {**c, "cosine": round(c["cosine"], 4),
                     "w_norm": round(c["w_norm"], 4),
                     "wprime_norm": round(c["wprime_norm"], 4),
                     "group": word_groups.get(c["word"], "other")}
                    for c in comparisons
                ],
            },
        }

        for name, data in outputs.items():
            path = out_dir / name
            with open(path, "w") as f:
                json.dump(data, f)
            sz = os.path.getsize(path)
            print(f"  {name:<30} {sz / 1024:>8.1f} KB")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
