#!/usr/bin/env python3
"""
Explore what your trained Word2Vec model knows
===============================================

Run this after train.py to interactively explore the learned embeddings.

Prints:
  1. Neighbor evolution — how each word's nearest neighbors change over training
  2. Analogy evolution — how analogy accuracy improves over training
  3. Rank evolution — how well the model predicts context words over time

Then (with --interactive) drops into a REPL where you can:
  - Type a word to see its nearest neighbors
  - Type an analogy like: king - man + woman
  - Type 'quit' to exit

Usage:
    python explore.py                        # reads from ./output/
    python explore.py --data-dir ./my_run
    python explore.py --interactive          # enter interactive mode after report
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KEY_WORDS = ["king", "queen", "cat", "dog", "man", "woman", "sun", "moon",
             "father", "mother", "lion", "prince", "happy", "boy", "girl"]

ANALOGIES = [
    ("king", "man", "woman", "queen"),
    ("queen", "woman", "man", "king"),
    ("prince", "boy", "girl", "princess"),
    ("father", "man", "woman", "mother"),
    ("son", "boy", "girl", "daughter"),
]

SAMPLE_STEPS = [0, 50, 100, 200, 500, 1000, 2000, 3000, 5000,
                7500, 10000, 12000, 15000]


# ── math helpers ─────────────────────────────────────────────────

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def k_nearest(word_idx, W, k=5):
    vec = W[word_idx]
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = W / norms
    vec_normed = vec / max(np.linalg.norm(vec), 1e-9)
    sims = normed @ vec_normed
    sims[word_idx] = -2.0
    top_idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_idx]


def solve_analogy(a_idx, b_idx, c_idx, W, exclude, k=3):
    v = W[a_idx] - W[b_idx] + W[c_idx]
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = W / norms
    v_normed = v / max(np.linalg.norm(v), 1e-9)
    sims = normed @ v_normed
    for idx in exclude:
        sims[idx] = -2.0
    top_idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_idx]


# ── evolution reports ────────────────────────────────────────────

def report_neighbor_evolution(data_dir, vocab, w2i, i2w, steps_to_process):
    print("=" * 60)
    print("1. NEIGHBOR EVOLUTION")
    print("=" * 60)
    print("  How each word's nearest neighbors change over training.\n")

    for word in KEY_WORDS:
        if word not in w2i:
            continue
        print(f"  {word}:")
        for step in steps_to_process:
            path = Path(data_dir) / "steps" / f"{step}.json"
            with open(path) as f:
                data = json.load(f)
            W = np.array(data["W"])
            neighbors = k_nearest(w2i[word], W, k=5)
            nbrs = "  ".join(f"{i2w[idx]}({sim:.2f})"
                             for idx, sim in neighbors)
            print(f"    step {step:>5}: {nbrs}")
        print()


def report_analogy_evolution(data_dir, vocab, w2i, i2w, steps_to_process):
    print("=" * 60)
    print("2. ANALOGY EVOLUTION")
    print("=" * 60)
    print("  How analogy accuracy improves over training.\n")

    print(f"  {'Step':>6}  ", end="")
    for a, b, c, expected in ANALOGIES:
        print(f"{a}-{b}+{c}={expected}?  ", end="")
    print(f"  {'Score':>5}")
    print(f"  {'─' * 6}  " + "─" * (len(ANALOGIES) * 22) + f"  {'─' * 5}")

    for step in steps_to_process:
        path = Path(data_dir) / "steps" / f"{step}.json"
        with open(path) as f:
            data = json.load(f)
        W = np.array(data["W"])

        correct = 0
        line = f"  {step:>6}  "
        for a, b, c, expected in ANALOGIES:
            if not all(w in w2i for w in (a, b, c, expected)):
                line += f"{'—':<22}"
                continue
            exclude = {w2i[a], w2i[b], w2i[c]}
            results = solve_analogy(w2i[a], w2i[b], w2i[c], W, exclude, k=1)
            predicted = i2w[results[0][0]]
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            mark = "✓" if is_correct else "✗"
            line += f"{predicted:<10} {mark}           "
        line += f"  {correct}/{len(ANALOGIES)}"
        print(line)


def report_rank_evolution(data_dir, steps_to_process):
    print("\n" + "=" * 60)
    print("3. CONTEXT PREDICTION QUALITY")
    print("=" * 60)
    print("  How well the model predicts context words over training.\n")

    loss_path = Path(data_dir) / "loss_curve.json"
    with open(loss_path) as f:
        loss_curve = json.load(f)

    print(f"  {'Step':>6}  {'Rank':>6}  {'P(ctx)':>8}  {'Avg Loss':>10}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 8}  {'─' * 10}")

    for step in steps_to_process:
        if step == 0:
            print(f"  {0:>6}  {'—':>6}  {'—':>8}  {'—':>10}")
            continue
        path = Path(data_dir) / "steps" / f"{step}.json"
        with open(path) as f:
            data = json.load(f)
        rank = data.get("context_rank", "—")
        prob = data.get("context_prob")
        window = [e for e in loss_curve if abs(e["step"] - step) <= 50]
        avg_loss = np.mean([e["loss"] for e in window]) if window else None

        rank_str = f"{rank:>6}" if isinstance(rank, int) else f"{'—':>6}"
        prob_str = f"{prob:>8.4f}" if prob is not None else f"{'—':>8}"
        loss_str = f"{avg_loss:>10.4f}" if avg_loss is not None else f"{'—':>10}"
        print(f"  {step:>6}  {rank_str}  {prob_str}  {loss_str}")


# ── interactive mode ─────────────────────────────────────────────

def interactive_mode(W, vocab, w2i, i2w):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("  Commands:")
    print("    <word>                     → nearest neighbors")
    print("    <a> - <b> + <c>            → word analogy")
    print("    quit                       → exit\n")

    while True:
        try:
            raw = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw or raw.lower() == "quit":
            break

        # Analogy: "king - man + woman"
        if " - " in raw and " + " in raw:
            parts = raw.replace("-", " - ").replace("+", " + ").split()
            tokens = [p.strip().lower() for p in parts if p.strip()
                      not in ("-", "+", "=", "?")]
            if len(tokens) == 3:
                a, b, c = tokens
                if all(w in w2i for w in (a, b, c)):
                    exclude = {w2i[a], w2i[b], w2i[c]}
                    results = solve_analogy(
                        w2i[a], w2i[b], w2i[c], W, exclude, k=5)
                    print(f"  {a} − {b} + {c} =")
                    for idx, sim in results:
                        print(f"    {i2w[idx]:<12} ({sim:.4f})")
                else:
                    missing = [w for w in (a, b, c) if w not in w2i]
                    print(f"  Unknown word(s): {', '.join(missing)}")
            else:
                print("  Format: word1 - word2 + word3")
            continue

        # Nearest neighbors
        word = raw.lower()
        if word in w2i:
            neighbors = k_nearest(w2i[word], W, k=10)
            print(f"  Nearest neighbors for '{word}':")
            for idx, sim in neighbors:
                print(f"    {i2w[idx]:<12} ({sim:.4f})")
        else:
            print(f"  '{word}' not in vocabulary. "
                  f"Try: {', '.join(vocab[:10])}...")


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Explore a trained Word2Vec model.")
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parent / "output"),
                    help="Path to training output (default: ./output/)")
    ap.add_argument("--interactive", action="store_true",
                    help="Enter interactive mode after the report")
    args = ap.parse_args()

    data_dir = args.data_dir
    meta_path = Path(data_dir) / "run_metadata.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run train.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    vocab = meta["vocab"]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}

    with open(Path(data_dir) / "step_index.json") as f:
        available_steps = set(json.load(f))

    steps_to_process = [s for s in SAMPLE_STEPS if s in available_steps]

    report_neighbor_evolution(data_dir, vocab, w2i, i2w, steps_to_process)
    report_analogy_evolution(data_dir, vocab, w2i, i2w, steps_to_process)
    report_rank_evolution(data_dir, steps_to_process)

    if args.interactive:
        final_step = max(available_steps)
        path = Path(data_dir) / "steps" / f"{final_step}.json"
        with open(path) as f:
            data = json.load(f)
        W = np.array(data["W"])
        interactive_mode(W, vocab, w2i, i2w)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
