"""Embedding-based near-duplicate clustering for mechanically-generated
pseudocode - genuinely ISA/language-agnostic (it only ever sees text), so
this lives in core rather than as a per-target plugin, unlike
MechanicalTranslator/TraceRanker.

Ported from jhonatanTeixeira/portal_to_another_world's
sandbox_hle/embed_cluster.py, itself the second iteration of a real,
same-day case study (2026-08-07) - keep both real findings from that study
in mind before changing the defaults below:

1. Raw cosine similarity from a base BERT-family model (no contrastive
   fine-tuning) is USELESS here - a textbook case of embedding anisotropy.
   Measured on a real 1106-proposal corpus: raw cosine ranged 0.80-0.9996
   for EVERY pair (mean 0.96) with microsoft/codebert-base - it cannot
   discriminate duplicates from unrelated code at all. Mean-centering
   (subtract the corpus mean embedding, renormalize) is the standard cheap
   fix and is applied unconditionally below, not an opt-in.
2. Model choice matters even after centering. microsoft/codebert-base gave
   sloppy separation (known duplicates 0.40-0.90 centered, a random-pair
   baseline overlapping up to 0.68). microsoft/graphcodebert-base (the
   default here) separated cleanly on the same corpus (duplicates -0.002 to
   0.58, random baseline ALL negative, -0.12 to -0.03). Still imperfect -
   treat every returned cluster as a candidate for the LLM-polish step (or a
   human) to double check, never as ground truth; a live run at threshold
   0.35 still produced one real false-positive merge of 3 unrelated
   functions, only caught by direct inspection - a stricter threshold
   (0.55) fixed that specific case but trades away some real recall.

Runs on CPU by default, not "cuda if available": in practice this competes
for the SAME small GPU the polishing LLM has already claimed most of (a
real torch.OutOfMemoryError was hit live sharing a 4GB card with a resident
3.4GB LLM). A few dozen-to-hundred short code snippets embed in low single
digits of seconds on CPU regardless - there is no real cost to just not
contending for the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusterResult:
    clusters: list[list[str]]
    pairs_above_threshold: list[tuple[str, str, float]]


def cluster_texts(
    texts: dict[str, str],
    model_id: str = "microsoft/graphcodebert-base",
    threshold: float = 0.55,
    device: str = "cpu",
) -> ClusterResult:
    """`texts`: {key: code_text}. Returns clusters of keys (every input key
    appears in exactly one cluster, including singletons) plus the raw
    above-threshold pairs for auditing. Requires torch+transformers - not a
    core dependency of this package (see pyproject.toml's [project.optional-
    dependencies] `clustering` group); raise a clear ImportError rather than
    letting the caller see a confusing failure deep in torch's own import
    machinery."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "semantic_clustering.cluster_texts needs torch+transformers - "
            "install with `pip install hle-studio[clustering]`"
        ) from e

    keys = list(texts.keys())
    if len(keys) < 2:
        return ClusterResult(clusters=[[k] for k in keys], pairs_above_threshold=[])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    values = [texts[k] for k in keys]
    embeddings = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            embeddings.append(F.normalize(summed / counts, p=2, dim=1).cpu())
    emb = torch.cat(embeddings, dim=0)

    # Anisotropy fix - see module docstring point 1. Not optional.
    mean_emb = emb.mean(dim=0, keepdim=True)
    centered = F.normalize(emb - mean_emb, p=2, dim=1)
    sim = centered @ centered.T

    n = len(keys)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim[i, j].item()
            if s >= threshold:
                union(i, j)
                pairs.append((keys[i], keys[j], round(s, 3)))

    groups: dict[int, list[str]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(find(i), []).append(k)

    return ClusterResult(clusters=list(groups.values()), pairs_above_threshold=pairs)
