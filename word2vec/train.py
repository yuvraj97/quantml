#!/usr/bin/env python3
"""
Word2Vec Skip-Gram — Full State Recording for Visualization
============================================================

Trains a Skip-Gram model on a curated corpus and records strategic
training steps' complete state for interactive visualization.

Architecture:
    Input (one-hot)  →  W (center embeddings, |V|×d)
                     →  v_c = W[center_word]
                     →  logits = W' @ v_c
                     →  softmax → P(·|center_word)

Loss:  L = -log P(context_word | center_word)   (cross-entropy)

Gradient decomposition for center embedding v_c:

    ∂L/∂v_c = −u_{w_o}  +  Σ_w P(w|w_c) u_w
              ─────────     ─────────────────
              attractive       repulsive

    update direction (−∂L/∂v_c) = u_{w_o} − E[u_w]
      → pull center toward true context
      → push center away from predicted distribution

Recording strategy:
    Steps [0, dense_steps):  every step  (early dynamics)
    Steps [dense_steps, N]:  every record_interval-th step

Output structure (local files, no DB):
    output/
    ├── run_metadata.json          vocab, config, word frequencies, raw sentences
    ├── loss_curve.json            all training steps' loss values
    ├── step_index.json            list of recorded step numbers
    ├── embeddings_timeline.json   2D projections at every recorded step
    └── steps/
        ├── 0.json                 full state per recorded step (~45 KB each)
        ├── 1.json
        └── ...

Usage:
    python train.py                          # output → ./output/
    python train.py --output-dir ./my_output
    python train.py --steps 25000 --lr 0.08
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

# ── stopwords ───────────────────────────────────────────────────────
STOPWORDS = frozenset({
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been",
    "being", "in", "on", "at", "to", "of", "and", "or", "but", "with",
    "by", "for", "from", "up", "down", "out", "into", "over", "under",
    "between", "his", "her", "its", "their", "he", "she", "it", "they",
    "we", "you", "i", "this", "that", "these", "those", "has", "had",
    "have", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "not", "no", "nor", "so", "very",
    "too", "also", "just", "then", "than", "as", "if", "when", "where",
    "how", "what", "which", "who", "whom", "there", "here", "all",
    "each", "every", "both", "few", "many", "much", "some", "any",
    "most", "other", "such", "only", "own", "same", "about", "after",
    "before", "while", "during", "through", "above", "below", "near",
})


# ── corpus ──────────────────────────────────────────────────────────

def load_corpus(filepath: str):
    raw, cleaned = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.lower().split()
            raw.append(tokens)
            content = [t for t in tokens if t not in STOPWORDS]
            if len(content) >= 2:
                cleaned.append(content)
    return raw, cleaned


def build_vocab(sentences, min_count=3):
    counts = Counter()
    for s in sentences:
        counts.update(s)
    vocab = sorted(w for w, c in counts.items() if c >= min_count)
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w, counts


def generate_pairs(sentences, w2i, window):
    pairs = []
    for sent in sentences:
        idxs = [w2i[w] for w in sent if w in w2i]
        for i, c in enumerate(idxs):
            lo, hi = max(0, i - window), min(len(idxs), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((c, idxs[j]))
    return pairs


# ── model ───────────────────────────────────────────────────────────

class SkipGram(nn.Module):
    def __init__(self, V: int, d: int):
        super().__init__()
        self.V, self.d = V, d
        self.W = nn.Embedding(V, d)
        self.W_prime = nn.Linear(d, V, bias=False)
        nn.init.uniform_(self.W.weight, -0.5, 0.5)
        nn.init.uniform_(self.W_prime.weight, -0.5, 0.5)

    def forward(self, center_idx: torch.Tensor):
        v_c = self.W(center_idx)
        logits = self.W_prime(v_c)
        return v_c.squeeze(0), logits.squeeze(0)


# ── state capture ──────────────────────────────────────────────────

def should_record(step, total, dense_steps, interval):
    if step <= dense_steps:
        return True
    if step % interval == 0:
        return True
    if step == total:
        return True
    return False


def capture_initial(model, i2w):
    """Step 0 — random initialisation, no training yet."""
    return {
        "step": 0,
        "center_word": None, "context_word": None, "loss": None,
        "W": model.W.weight.detach().cpu().numpy().tolist(),
        "W_prime": model.W_prime.weight.detach().cpu().numpy().tolist(),
        "hidden": None, "logits": None, "probabilities": None,
        "grad_v_c": None, "grad_u_o": None,
        "attractive_force": None, "repulsive_force": None,
        "cosine_center_context": None, "context_rank": None,
        "context_prob": None,
    }


def capture_step(step, ci, oi, model, loss_val, v_c, logits, probs, i2w):
    V = model.V
    W_np = model.W.weight.detach().cpu().numpy()
    Wp_np = model.W_prime.weight.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    hidden_np = v_c.detach().cpu().numpy()

    grad_v_c = model.W.weight.grad[ci].cpu().numpy()
    grad_u_o = model.W_prime.weight.grad[oi].cpu().numpy()

    u_o = Wp_np[oi]
    attractive = u_o.copy()
    repulsive = probs_np @ Wp_np

    sorted_idx = np.argsort(probs_np)[::-1]
    rank = int(np.where(sorted_idx == oi)[0][0]) + 1

    cos = float(F.cosine_similarity(
        torch.from_numpy(hidden_np).unsqueeze(0),
        torch.from_numpy(W_np[oi]).unsqueeze(0),
    ))

    return {
        "step": step,
        "center_word": i2w[ci],
        "context_word": i2w[oi],
        "loss": float(loss_val),
        "W": W_np.tolist(),
        "W_prime": Wp_np.tolist(),
        "hidden": hidden_np.tolist(),
        "logits": {i2w[i]: float(logits_np[i]) for i in range(V)},
        "probabilities": {i2w[i]: float(probs_np[i]) for i in range(V)},
        "grad_v_c": grad_v_c.tolist(),
        "grad_u_o": grad_u_o.tolist(),
        "attractive_force": attractive.tolist(),
        "repulsive_force": repulsive.tolist(),
        "cosine_center_context": cos,
        "context_rank": rank,
        "context_prob": float(probs_np[oi]),
    }


# ── training loop ──────────────────────────────────────────────────

def train(model, pairs, i2w, steps, lr, dense_steps, record_interval):
    loss_fn = nn.CrossEntropyLoss()
    random.shuffle(pairs)
    idx = 0

    records = []
    W_snapshots = []

    W0 = model.W.weight.detach().cpu().numpy()
    records.append(capture_initial(model, i2w))
    W_snapshots.append(W0.copy())

    all_losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        if idx >= len(pairs):
            random.shuffle(pairs)
            idx = 0
        ci, oi = pairs[idx]
        idx += 1

        ct, ot = torch.tensor([ci]), torch.tensor([oi])
        model.zero_grad()
        v_c, logits = model(ct)
        loss = loss_fn(logits.unsqueeze(0), ot)
        loss.backward()

        loss_val = loss.item()
        all_losses.append(loss_val)

        record = should_record(step, steps, dense_steps, record_interval)
        if record:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
            rec = capture_step(step, ci, oi, model, loss_val,
                               v_c, logits, probs, i2w)
            records.append(rec)
            W_snapshots.append(model.W.weight.detach().cpu().numpy().copy())

        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

        if step % 2000 == 0 or step == steps:
            elapsed = time.time() - t0
            avg = np.mean(all_losses[-500:]) if len(all_losses) >= 500 else np.mean(all_losses)
            print(f"  step {step:>6}/{steps}  "
                  f"avg_loss(500)={avg:.4f}  [{elapsed:.1f}s]  "
                  f"({len(records)} recorded)")

    return records, W_snapshots, all_losses


# ── projections ─────────────────────────────────────────────────────

def add_projections(records, W_snapshots, i2w, method):
    V = W_snapshots[0].shape[0]
    pca = PCA(n_components=2, random_state=42)
    if method == "final_pca":
        pca.fit(W_snapshots[-1])
    else:
        pca.fit(np.vstack(W_snapshots))

    var = pca.explained_variance_ratio_.tolist()
    for i, rec in enumerate(records):
        proj = pca.transform(W_snapshots[i])
        rec["embeddings_2d"] = {i2w[j]: proj[j].tolist() for j in range(V)}
        rec["pca_variance_explained"] = var


# ── storage ─────────────────────────────────────────────────────────

def save_local(records, all_losses, vocab, counts, config, raw_sentences,
               out_dir):
    """Write all training data as structured local files.

    Layout designed for fast frontend access:
      - run_metadata.json         loaded once on page init
      - loss_curve.json           loaded once for the chart
      - step_index.json           loaded once, tells frontend which steps exist
      - embeddings_timeline.json  loaded once for the scatter animation
      - steps/{N}.json            loaded on-demand when user inspects a step
    """
    steps_dir = os.path.join(out_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    # ── run_metadata.json (lightweight, ~50 KB) ────────────────
    meta = {
        "topic": "word2vec",
        "corpus_name": "curated_v1",
        "vocab": vocab,
        "vocab_size": len(vocab),
        "embed_dim": config["embed_dim"],
        "config": config,
        "total_steps": config["num_steps"],
        "recorded_steps": len(records),
        "word_frequencies": {w: counts[w] for w in vocab},
        "raw_sentences": [" ".join(s) for s in raw_sentences],
    }
    _write(os.path.join(out_dir, "run_metadata.json"), meta, indent=2)

    # ── loss_curve.json (every training step, ~300 KB) ─────────
    loss_curve = [{"step": i + 1, "loss": round(l, 6)}
                  for i, l in enumerate(all_losses)]
    _write(os.path.join(out_dir, "loss_curve.json"), loss_curve)

    # ── step_index.json (which steps have full recordings) ─────
    step_index = [rec["step"] for rec in records]
    _write(os.path.join(out_dir, "step_index.json"), step_index)

    # ── embeddings_timeline.json (~3 MB) ───────────────────────
    # Compact: just step number → {word: [x, y]} for the scatter
    # animation. Frontend loads this once and scrubs through it.
    timeline = []
    for rec in records:
        entry = {
            "step": rec["step"],
            "positions": rec["embeddings_2d"],
        }
        if rec["loss"] is not None:
            entry["loss"] = round(rec["loss"], 6)
            entry["center_word"] = rec["center_word"]
            entry["context_word"] = rec["context_word"]
        timeline.append(entry)
    _write(os.path.join(out_dir, "embeddings_timeline.json"), timeline)

    # ── steps/{N}.json (one file per recorded step, ~45 KB each)
    total_step_bytes = 0
    for rec in records:
        step_path = os.path.join(steps_dir, f"{rec['step']}.json")
        step_data = {k: v for k, v in rec.items()
                     if k != "embeddings_2d" and k != "pca_variance_explained"}
        step_data["embeddings_2d"] = rec["embeddings_2d"]
        step_data["pca_variance_explained"] = rec.get("pca_variance_explained")
        _write(step_path, step_data)
        total_step_bytes += os.path.getsize(step_path)

    # ── report ─────────────────────────────────────────────────
    sizes = {}
    for name in ["run_metadata.json", "loss_curve.json",
                 "step_index.json", "embeddings_timeline.json"]:
        p = os.path.join(out_dir, name)
        sizes[name] = os.path.getsize(p)

    print(f"  Output directory: {out_dir}/")
    for name, sz in sizes.items():
        print(f"    {name:<30} {sz / 1024:>8.1f} KB")
    print(f"    steps/ ({len(records)} files){' ' * 14}"
          f"{total_step_bytes / 1e6:>7.1f} MB")
    print(f"    {'TOTAL':<30} "
          f"{(sum(sizes.values()) + total_step_bytes) / 1e6:>7.1f} MB")


def _write(path, data, indent=None):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


# ── summary ─────────────────────────────────────────────────────────

def print_summary(model, w2i, i2w):
    W = model.W.weight.detach()

    print("\n  Nearest neighbors (cosine, center embeddings):")
    for word in ["king", "queen", "cat", "dog", "man", "woman",
                 "lion", "wolf", "sun", "moon", "father", "mother",
                 "happy", "prince", "boy"]:
        if word not in w2i:
            continue
        sims = F.cosine_similarity(W[w2i[word]].unsqueeze(0), W)
        sims[w2i[word]] = -2
        top = torch.argsort(sims, descending=True)[:5]
        nbrs = "  ".join(f"{i2w[j.item()]}({sims[j]:.2f})" for j in top)
        print(f"    {word:>10} → {nbrs}")

    print("\n  Analogies (a − b + c = ?):")
    for a, b, c, expect in [
        ("king", "man", "woman", "queen"),
        ("queen", "woman", "man", "king"),
        ("prince", "boy", "girl", "princess"),
        ("father", "man", "woman", "mother"),
        ("son", "boy", "girl", "daughter"),
    ]:
        if not all(w in w2i for w in (a, b, c)):
            continue
        v = W[w2i[a]] - W[w2i[b]] + W[w2i[c]]
        sims = F.cosine_similarity(v.unsqueeze(0), W)
        for w in (a, b, c):
            sims[w2i[w]] = -2
        top3 = torch.argsort(sims, descending=True)[:3]
        results = "  ".join(f"{i2w[j.item()]}({sims[j]:.2f})" for j in top3)
        mark = " ✓" if i2w[top3[0].item()] == expect else ""
        print(f"    {a} − {b} + {c} = {results}{mark}")


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(Path(__file__).with_name("corpus.txt")))
    ap.add_argument("--output-dir",
                    default=str(Path(__file__).resolve().parent / "output"),
                    help="Where to write output files (default: ./output/)")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense-steps", type=int, default=200,
                    help="Record every step for the first N steps")
    ap.add_argument("--record-interval", type=int, default=10,
                    help="After dense-steps, record every N-th step")
    ap.add_argument("--pca", default="final_pca",
                    choices=["final_pca", "global_pca"])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── load ────────────────────────────────────────────────────
    print("Loading corpus …")
    raw, cleaned = load_corpus(args.corpus)
    vocab, w2i, i2w, counts = build_vocab(cleaned, args.min_count)
    pairs = generate_pairs(cleaned, w2i, args.window)

    print(f"  {len(raw)} sentences  →  {len(cleaned)} after stopword removal")
    print(f"  Vocabulary: {len(vocab)} words")
    print(f"  Training pairs: {len(pairs)}")
    print(f"  Epochs covered: {args.steps / len(pairs):.1f}")

    config = {
        "learning_rate": args.lr,
        "window_size": args.window,
        "embed_dim": args.dim,
        "num_steps": args.steps,
        "min_count": args.min_count,
        "random_seed": args.seed,
        "dense_steps": args.dense_steps,
        "record_interval": args.record_interval,
        "pca_method": args.pca,
    }

    # ── train ───────────────────────────────────────────────────
    model = SkipGram(len(vocab), args.dim)
    print(f"\nTraining (steps={args.steps}, dim={args.dim}, lr={args.lr}) …")
    records, W_snaps, all_losses = train(
        model, pairs, i2w, args.steps, args.lr,
        args.dense_steps, args.record_interval,
    )

    # ── 2D projections ─────────────────────────────────────────
    print(f"\nProjecting to 2D ({args.pca}, {len(records)} snapshots) …")
    add_projections(records, W_snaps, i2w, args.pca)

    # ── save ────────────────────────────────────────────────────
    print(f"\nSaving to {args.output_dir} …")
    save_local(records, all_losses, vocab, counts, config, raw, args.output_dir)

    # ── summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total steps trained : {args.steps}")
    print(f"  Steps recorded      : {len(records)}")
    print(f"  Vocab size          : {len(vocab)}")
    print(f"  Embed dim           : {args.dim}")
    print(f"  Initial loss        : {all_losses[0]:.4f}")
    print(f"  Final loss (avg 500): {np.mean(all_losses[-500:]):.4f}")
    print_summary(model, w2i, i2w)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
