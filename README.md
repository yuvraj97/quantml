# QuantML Learn

Implementations behind [quantml.org](https://www.quantml.org) — every model taught on the site, built from scratch in clean Python.

Each directory is a self-contained project with its own `requirements.txt`, training script, and analysis tools. Clone the repo, pick a topic, and follow its README.

## Topics

| Topic | What you'll build | Interactive lesson |
|-------|-------------------|--------------------|
| [Word2Vec](./word2vec/) | Skip-gram word embeddings in PyTorch | [quantml.org/topics/word2vec](https://www.quantml.org/topics/word2vec) |

## Quick Start

```bash
git clone https://github.com/quantml/learn.git
cd learn/word2vec
pip install -r requirements.txt
python train.py
```

## How This Repo Relates to the Website

The [QuantML website](https://www.quantml.org) provides interactive visualizations, guided walkthroughs, and quizzes for each topic. This repo contains the actual training code that generates the data behind those visualizations. You can:

1. **Read the code** to understand the implementation
2. **Run it yourself** to reproduce results
3. **Modify hyperparameters** and experiment
4. **Analyze the outputs** with the included analysis scripts

## License

[MIT](./LICENSE)
