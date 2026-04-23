# Darija N-Gram Language Model

A probabilistic trigram language model built from scratch on a
multi-source Moroccan Darija corpus, no external NLP libraries used

## Features (applied strategies we only saw in class)
- Unigram, Bigram, Trigram frequency counting
- Laplace (Add-1) smoothing
- Stupid Backoff strategy (trigram → bigram → unigram)
- Sentence log-probability and perplexity
- Darija sentence generation via weighted sampling
- Frequency distribution visualizations

## Project Structure
```
data/
├── darija_ngram_lm.ipynb
├──ngram_model_results.json
├──ngram_frequencies.png
└── README.md
```

## Data
The corpus is not included in this repository due to file size.
It contains the following sources:

| Source       | Content                        |
|--------------|--------------------------------|
| darija-wiki  | Wikipedia articles in Darija   |
| goud.ma      | News articles                  |
| Twitter      | tweets                         |
| story-data   | Fiction stories                |
| music-data   | Song lyrics                    |
| Youtube      | YouTube comments               |

To use this project, place your data folder at `data/` in the
project root, matching the structure shown above.

## Setup

```bash
git clone https://github.com/your-username/darija-ngram-lm.git
cd darija-ngram-lm
pip install -r requirements.txt
jupyter notebook notebooks/darija_ngram_lm.ipynb
```

## How It Works

### Smoothed Probabilities
```
P_smooth(w)         = (C(w) + 1)         / (N + |V|)
P_smooth(w₂|w₁)     = (C(w₁,w₂) + 1)    / (C(w₁) + |V|)
P_smooth(w₃|w₁,w₂) = (C(w₁,w₂,w₃) + 1) / (C(w₁,w₂) + |V|)
```

### Backoff Strategy
```
if C(w₁,w₂,w₃) > 0  →  use trigram probability
elif C(w₂,w₃)  > 0  →  λ  × bigram probability
else                 →  λ² × unigram probability
```

### Perplexity
```
PP = exp( -1/T · Σ log P(wᵢ | context) )
```
