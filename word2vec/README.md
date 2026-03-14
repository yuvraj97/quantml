# Word2Vec — Skip-Gram from Scratch

Train a Word2Vec model from scratch in PyTorch and watch words learn to mean something.

> **Interactive lesson:** [quantml.org/topics/word2vec](https://www.quantml.org/topics/word2vec)
> — every concept here has an interactive visual twin on the website.

## What You'll Learn

- How the **skip-gram** architecture predicts context words from a center word
- How **gradient descent** shapes the embedding space — the attractive/repulsive force decomposition
- How to **inspect trained embeddings** — similarity matrices, convergence curves, W vs W'
- How to **test with word analogies** — king - man + woman = queen

## Architecture

```
Input (one-hot)  →  W (center embeddings, |V|×d)
                 →  v_c = W[center_word]
                 →  logits = W' @ v_c
                 →  softmax → P(·|center_word)

Loss:  L = -log P(context_word | center_word)   (cross-entropy)

Gradient decomposition for center embedding v_c:

    ∂L/∂v_c = −u_{w_o}  +  Σ_w P(w|w_c) u_w
              ─────────     ─────────────────
              attractive       repulsive

    → pull center toward true context
    → push center away from predicted distribution
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py              # trains in ~30s, outputs to ./output/
python analyze.py            # inspect what the model learned
python explore.py            # neighbor + analogy evolution over training
python explore.py --interactive   # query the model yourself
```

## Expected Output

After training completes, you'll see results like:

```
Nearest neighbors (cosine, center embeddings):
      king → queen(0.91)  prince(0.88)  throne(0.84)  crown(0.83)  kingdom(0.82)
     queen → king(0.91)  princess(0.89)  prince(0.85)  throne(0.83)  crown(0.82)
       cat → dog(0.93)  rabbit(0.88)  bird(0.85)  fish(0.83)  animal(0.82)
       man → woman(0.93)  boy(0.89)  father(0.87)  son(0.85)  girl(0.83)

Analogies (a − b + c = ?):
    king − man + woman = queen ✓
    prince − boy + girl = princess ✓
    father − man + woman = mother ✓
    son − boy + girl = daughter ✓
```

> This is a **tiny toy training run** — 96 words, 8 dimensions, 15,000 steps.
> It exists to teach how Word2Vec works, not to produce production embeddings.
> Google's original Word2Vec used 3M words and 300 dimensions.

## File Guide

| File | What it does |
|------|-------------|
| `train.py` | Core skip-gram training with full state recording. 484 lines, heavily commented. |
| `corpus.txt` | 545 hand-crafted sentences with deliberate parallel structures (king/queen, man/woman, etc.) so analogies emerge from a tiny corpus. |
| `analyze.py` | Post-training analysis: similarity matrix, convergence curves, W vs W' comparison, corpus stats. Run with `--save-json` to export results. |
| `explore.py` | Evolution reports (how neighbors and analogies change during training) + interactive REPL mode for querying the model. |
| `requirements.txt` | Python dependencies (PyTorch, NumPy, scikit-learn). |
| `sample_output/` | Pre-generated output files from a reference run so you can browse expected results without training first. |

## How the Corpus Was Designed

The 545 sentences in `corpus.txt` are hand-written with deliberate parallel structures:

- **"The king rules the kingdom"** / **"The queen rules the kingdom"** — king and queen appear in identical contexts, forcing the model to give them similar embeddings
- **"The boy plays in the garden"** / **"The girl plays in the garden"** — same pattern for gendered pairs
- **"The cat sits on the mat"** / **"The dog sits on the mat"** — animals in parallel contexts

This is why analogies work even with only 96 words — the corpus encodes the relationships we want the model to discover.

## Experiments to Try

### Change embedding dimension

```bash
python train.py --dim 4     # fewer dimensions — harder to separate 96 words
python train.py --dim 16    # more capacity — easier but harder to visualize
```

Fewer dimensions = less capacity but faster training. 4 dimensions may struggle with 96 words; 16 should do better.

### Change learning rate

```bash
python train.py --lr 0.01   # too slow — barely learns in 15K steps
python train.py --lr 0.5    # too fast — loss may diverge
```

0.1 is the sweet spot for this corpus size. Try extremes to see what happens.

### Change window size

```bash
python train.py --window 2  # smaller — more syntactic similarity
python train.py --window 5  # larger — more semantic/topical similarity
```

Larger windows capture broader context (topical relatedness). Smaller windows capture immediate neighbors (syntactic roles).

### Change training steps

```bash
python train.py --steps 5000    # early — some structure, weak analogies
python train.py --steps 25000   # longer — diminishing returns
```

Watch diminishing returns: loss drops fast in the first 5K, then plateaus. More steps = better analogies but slower.

## Output Structure

After running `train.py`, you'll find:

```
output/
├── run_metadata.json          vocab, config, word frequencies, raw sentences
├── loss_curve.json            all 15,000 steps' loss values
├── step_index.json            list of recorded step numbers
├── embeddings_timeline.json   2D projections at every recorded step
└── steps/
    ├── 0.json                 full state at step 0 (~45 KB)
    ├── 1.json                 full state at step 1
    └── ...                    1,681 step files total (~73 MB)
```

Each step file contains: W matrix, W' matrix, hidden vector, logits, probabilities, gradients, attractive/repulsive forces, cosine similarity, context rank.

## Interactive Lesson

This code is the engine behind the interactive lesson at [quantml.org](https://www.quantml.org/topics/word2vec). The website provides:

| Tab | What you'll find | Link |
|-----|-----------------|------|
| The Story | Visual, chapter-by-chapter introduction to Word2Vec concepts | [quantml.org/topics/word2vec](https://www.quantml.org/topics/word2vec) |
| Inside the Model | Weight matrix animations, similarity heatmaps, forward/backward pass walkthrough | [quantml.org/topics/word2vec/internals](https://www.quantml.org/topics/word2vec/internals) |
| The Code | Annotated walkthrough of this exact training script with 64 inline annotations | [quantml.org/topics/word2vec/code](https://www.quantml.org/topics/word2vec/code) |
| Quiz | 15 questions across 3 difficulty tiers to test your understanding | [quantml.org/topics/word2vec/quiz](https://www.quantml.org/topics/word2vec/quiz) |

## License

[MIT](../LICENSE)
