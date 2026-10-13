"""
Feature extraction for the Human vs AI Writing Pattern Analyzer.
Created by the training notebook. Keep this file next to app.py.
"""
import re
from collections import Counter

import nltk
import numpy as np
import textstat

for _res, _pkg in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab"),
                   ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
                   ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)

CHUNK_WORDS = 150      # every text is analysed in sections of about this many words

WORD_RE        = re.compile(r"[A-Za-z']+")
CONTRACTION_RE = re.compile(r"\b\w+(?:n't|'re|'ve|'ll|'m|'d)\b", re.I)
PASSIVE_RE     = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", re.I)
CITATION_RE    = re.compile(r"\[(?:\w{1,4}|citation needed)\](?::\d+(?:[–-]\d+)?)?", re.I)
FIRST_PERSON   = {"i", "me", "my", "mine", "we", "us", "our", "ours"}


def clean_text(text):
    text = CITATION_RE.sub("", text)                  # citation marks like [a], [12], [5]:10–11
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # single line break = wrapped line
    return text.strip()


def cv(values):
    """Coefficient of variation = std / mean (burstiness)."""
    values = np.asarray(values, dtype=float)
    return values.std() / values.mean() if len(values) > 1 and values.mean() > 0 else 0.0


def repeat_ratio(tokens, n):
    """Share of n-grams that occur more than once."""
    grams = list(zip(*[tokens[i:] for i in range(n)]))
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / len(grams)


def split_chunks(text, size=CHUNK_WORDS, min_last=100):
    """Cut a text into consecutive sections of ~size words, always ending at a sentence end.
    A leftover shorter than min_last words is merged into the previous section."""
    sents = nltk.sent_tokenize(clean_text(text))
    chunks, cur, n = [], [], 0
    for s in sents:
        cur.append(s)
        n += len(WORD_RE.findall(s))
        if n >= size:
            chunks.append((" ".join(cur), n))
            cur, n = [], 0
    if cur:
        if chunks and n < min_last:
            prev, pn = chunks[-1]
            chunks[-1] = (prev + " " + " ".join(cur), pn + n)
        else:
            chunks.append((" ".join(cur), n))
    return [c for c, _ in chunks] or [clean_text(text)]


def text_stats(text):
    """Basic counts used for input validation and charts."""
    text = clean_text(text)
    sentences = nltk.sent_tokenize(text) if text else []
    return {
        "n_words": len(WORD_RE.findall(text)),
        "n_sentences": len(sentences),
        "sentence_lengths": [len(WORD_RE.findall(s)) for s in sentences],
    }


def paragraph_stats(text):
    paras = [p for p in re.split(r"\n\s*\n", clean_text(text)) if p.strip()]
    lens = [len(WORD_RE.findall(p)) for p in paras]
    return {"n_paragraphs": len(paras), "avg_para_len": float(np.mean(lens)) if lens else 0.0}


def repeated_phrases(text, n=2, top=8):
    """Most repeated n-grams (ignoring ones made only of very common words)."""
    stop = {"the", "a", "an", "of", "to", "and", "in", "is", "it", "that", "for", "on", "as", "with", "be"}
    tokens = [w.lower() for w in WORD_RE.findall(clean_text(text))]
    grams = [" ".join(g) for g in zip(*[tokens[i:] for i in range(n)])
             if not all(w in stop for w in g)]
    return [(g, c) for g, c in Counter(grams).most_common(top) if c > 1]


def extract_features(text):
    text       = clean_text(text)
    sentences  = nltk.sent_tokenize(text)
    words      = WORD_RE.findall(text)
    lower      = [w.lower() for w in words]
    n_words    = max(len(words), 1)
    n_sents    = max(len(sentences), 1)
    sent_lens  = [len(WORD_RE.findall(s)) for s in sentences] or [0]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    para_lens  = [len(WORD_RE.findall(p)) for p in paragraphs]

    tags = Counter(tag[:2] for _, tag in nltk.pos_tag(words))
    pct  = lambda k: 100 * tags[k] / n_words
    noun, verb, adj, adv, pron, prep, det = (pct(k) for k in ["NN", "VB", "JJ", "RB", "PR", "IN", "DT"])

    return {
        "avg_sent_len": np.mean(sent_lens), "std_sent_len": np.std(sent_lens),
        "min_sent_len": np.min(sent_lens),  "max_sent_len": np.max(sent_lens),
        "ttr": len(set(lower[:200])) / max(len(lower[:200]), 1),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "long_word_pct": 100 * sum(len(w) >= 7 for w in words) / n_words,
        "bigram_repeat": repeat_ratio(lower, 2), "trigram_repeat": repeat_ratio(lower, 3),
        "burstiness": cv(sent_lens),
        "formality": (noun + adj + prep + det - pron - verb - adv + 100) / 2,
        "contraction_rate": 100 * len(CONTRACTION_RE.findall(text)) / n_words,
        "first_person_rate": 100 * sum(w in FIRST_PERSON for w in lower) / n_words,
        "passive_rate": 100 * len(PASSIVE_RE.findall(text)) / n_sents,
        "n_paragraphs": len(paragraphs),
        "avg_para_len": np.mean(para_lens) if para_lens else 0,
        "para_len_cv": cv(para_lens),
        "flesch_ease": textstat.flesch_reading_ease(text),
        "fk_grade": textstat.flesch_kincaid_grade(text),
        "noun_pct": noun, "verb_pct": verb, "adj_pct": adj,
        "adv_pct": adv, "pron_pct": pron, "prep_pct": prep,
        "comma_rate": text.count(",") / n_sents,
        "question_rate": text.count("?") / n_sents,
    }
