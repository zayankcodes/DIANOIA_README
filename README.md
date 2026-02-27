# DIANOIA

**DIANOIA** is an end-to-end argument analysis system that turns raw argumentative text into structured argument graphs, diagnoses structural and semantic inconsistencies, proposes minimal repairs, detects fallacies, and computes formal argumentation semantics.

At its core, the project enforces a single internal representation for every stage of the pipeline: corpus ingestion, graph construction, NLI-based coherence checking, repair, semantics, explanation, evaluation, API serving, and CLI tooling.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Data Model](#core-data-model)
3. [System Architecture](#system-architecture)
4. [Corpora and Conversion](#corpora-and-conversion)
5. [Graph Diagnostics and Repair](#graph-diagnostics-and-repair)
6. [Argumentation Semantics](#argumentation-semantics)
7. [Raw-Text Extraction](#raw-text-extraction)
8. [Engine Orchestration](#engine-orchestration)
9. [Explanations and Evaluation](#explanations-and-evaluation)
10. [API](#api)
11. [CLI](#cli)
12. [Training](#training)
13. [End-to-End Request Flow](#end-to-end-request-flow)
14. [Design Principles](#design-principles)

---

## Overview

The codebase is organized as a layered pipeline:

- **Canonical data layer** for ADUs, edges, and documents
- **Corpus converters** that normalize heterogeneous datasets into one schema
- **Graph layer** built on NetworkX
- **Sanity checks** for structural pathologies
- **NLI scoring** for semantic agreement between annotated relations and text content
- **Coherence diagnostics** that flag label/content mismatches
- **Witness generation** that suggests minimal repairs
- **Repair operations** that transform a graph without mutating the original
- **Fallacy detection** via topology and optional learned classifiers
- **Argumentation semantics** for acceptability and graded strength
- **Raw-text extraction** for direct analysis without gold annotations
- **Engine, API, CLI, and evaluation tooling** to run the full system in practice

The guiding idea is simple: **every dataset is normalized into the same canonical representation, and every downstream component consumes that representation or the graph derived from it.**

---

## Core Data Model

Everything in DIANOIA rests on three frozen dataclasses defined in `types.py`.

### `CanonADU`
A single Argument Discourse Unit (ADU): claim, premise, major claim, or other discourse unit.

```python
CanonADU(
    id="a1",
    span=(0, 47),
    text="Climate change is real.",
    type="major_claim",
    attrs={},
)
```

Fields:
- `id`: unique ADU identifier within a document
- `span`: character offsets into the original document text
- `text`: extracted ADU text
- `type`: discourse role such as `major_claim`, `claim`, or `premise`
- `attrs`: corpus-specific metadata such as confidence or raw labels

### `CanonEdge`
A directed argumentative relation between two ADUs.

```python
CanonEdge(
    id="r1",
    src="a1",
    tgt="a2",
    label="support",
    attrs={"label_raw": "sup"},
)
```

Fields:
- `id`: unique edge identifier
- `src`: source ADU id
- `tgt`: target ADU id
- `label`: normalized to `support`, `attack`, or `other`
- `attrs`: auxiliary metadata, including the raw corpus label when relevant

### `CanonDoc`
The full canonical document object.

Fields:
- `doc_id`
- `corpus`
- `split`
- `text`
- `adus: list[CanonADU]`
- `edges: list[CanonEdge]`

`CanonDoc` is the universal currency of the system: **all converters produce it and all downstream processing begins from it.**

### Validation
`CanonDoc.validate()` enforces the basic invariants:

- document text must be non-empty
- ADU ids must be unique
- edge ids must be unique
- every edge must reference existing ADU ids

Invalid data is rejected early rather than propagating downstream.

---

## System Architecture

The repository is easiest to understand as a sequence of layers.

### Layer 0 - Data Types
**File:** `types.py`

Defines `CanonADU`, `CanonEdge`, and `CanonDoc`.

### Layer 1 - Corpus Converters
**Directory:** `datasets/`

Each corpus loader converts raw annotations into `CanonDoc` objects and writes JSONL.

### Layer 2 - I/O
**File:** `io/jsonl.py`

Provides JSONL serialization and deserialization with validation on both write and read.

### Layer 3 - Graph Construction
**File:** `graph/build.py`

Transforms a `CanonDoc` into an `nx.DiGraph`.

### Layer 4 - Structural Sanity
**File:** `graph/sanity.py`

Flags graph-level pathologies such as isolated nodes, cycles, and disconnected components.

### Layer 5 - NLI Model
**File:** `nlp/nli.py`

Wraps a cross-encoder NLI model behind a protocol interface.

### Layer 6 - Coherence Checking
**File:** `diagnostics/coherence.py`

Measures whether the semantic content of an edge matches its annotated label.

### Layer 7 - Witnesses
**File:** `diagnostics/witnesses.py`

Turns coherence failures into repair hints such as label flips or deletions.

### Layer 8 - Repair
**File:** `repair/ops.py`

Applies graph edits on copies and records an audit trail.

### Layer 9 - Fallacy Detection
**File:** `diagnostics/fallacies.py`

Detects fallacies from both graph topology and optional learned models.

### Layer 10 - Argumentation Semantics
**File:** `graph/semantics.py`

Computes Dung-style extensions and graded strength metrics.

### Layer 11 - Extraction
**File:** `nlp/extract.py`

Builds synthetic `CanonDoc` objects directly from raw text.

### Layer 12 - Engine
**File:** `engine.py`

Orchestrates the end-to-end pipeline and returns an `AnalysisResult`.

### Layer 13 - Explanations
**Directory:** `explain/`

Converts analysis results into bullets, narrative prose, and structured reports.

### Layer 14 - Evaluation
**File:** `eval/metrics.py`

Implements benchmark metrics for coherence, repair, fallacy detection, and extraction.

### Layer 15 - API
**Directory:** `api/`

Exposes the engine over FastAPI.

### Layer 16 - CLI
**File:** `cli/build_data.py`

Provides dataset-building utilities from the command line.

### Layer 17 - Training
**File:** `scripts/finetune_nli.py`

Fine-tunes the NLI model on argument-relation pairs.

---

## Corpora and Conversion

DIANOIA currently supports three corpora.

| Corpus | Files | Splits | Notes |
|---|---|---:|---|
| **ArgMicro** | `datasets/argmicro.py` | train only | 112 German + 112 English microtexts; XML-based; EDUs are merged into ADUs via segmentation edges |
| **AAE v2** | `datasets/aae2.py` | train/dev/test | 402 student essays; main evaluation corpus because it has proper dev/test splits |
| **CDCP** | `datasets/cdcp.py` | train/test | Cornell eRulemaking corpus; support-only relations; useful for augmenting entailment during NLI fine-tuning |

### ArgMicro
ArgMicro consists of paired `.txt` and `.xml` files.

The converter:
1. parses XML with `ElementTree`
2. reconstructs ADUs by merging EDUs connected through `seg` edges
3. collects non-seg inter-ADU edges
4. normalizes labels via `_EDGE_MAP`
5. emits a `CanonDoc`

Relation normalization:
- `sup`, `exa`, `add` -> `support`
- `reb`, `und` -> `attack`

Because the HuggingFace loader was incompatible with newer `datasets` versions, the builder downloads the corpus zip directly from GitHub.

### AAE v2
The Argument Annotated Essays v2 corpus is loaded from HuggingFace.

Properties:
- 402 student essays
- splits: 321 train / 40 dev / 41 test
- ADU types include `major_claim`, `claim`, and `premise`
- edges include `support` and `attack`

This is the principal evaluation benchmark for the full pipeline.

### CDCP
CDCP uses paired `.txt` and `.ann.json` files.

The annotation JSON includes:
- `prop_offsets`
- `prop_labels`
- `reasons`
- `evidences`

In DIANOIA, these are converted into ADUs plus directed support edges. CDCP contains **no attack edges**, which makes it especially useful for strengthening the entailment class when fine-tuning the NLI component.

---

## I/O and Validation

**File:** `io/jsonl.py`

Two functions define the persistence layer.

### `write_jsonl(path, docs)`
- serializes a sequence of `CanonDoc` objects to JSONL
- uses `dataclasses.asdict()` before `json.dumps`
- validates every document before writing

### `read_jsonl(path)`
- reads JSONL back into `CanonDoc` objects
- validates each document on load
- raises clear line-numbered errors for malformed JSON

The design goal is explicit: **corrupt data should fail immediately at the I/O boundary, not halfway through the pipeline.**

---

## Graph Diagnostics and Repair

### Graph Construction
**File:** `graph/build.py`

`build_graph(doc)` converts a `CanonDoc` into a directed NetworkX graph.

Each ADU becomes a node with attributes:
- `text`
- `type`
- `span`
- `attrs`

Each edge becomes a directed graph edge with attributes:
- `id`
- `label`
- `weight` (default `1.0`, later overwritten by NLI)
- `attrs`

Graph-level metadata such as `doc_id` and `corpus` is stored on the graph object itself.

Once the graph is built, the rest of the pipeline works on the graph rather than the original `CanonDoc`.

### Structural Sanity Checks
**File:** `graph/sanity.py`

Three structural warnings are computed without any model calls:

- **`ISOLATED_NODE`**: node degree is zero
- **`CYCLE`**: directed cycles found via `nx.simple_cycles(g)`
- **`DISCONNECTED`**: graph is not weakly connected

Utility:
- `roots(g)` returns nodes with no incoming edges

### NLI Scoring
**File:** `nlp/nli.py`

The NLI layer defines a common protocol:

- `score(premise, hypothesis) -> NLIScores`
- `batch_score(pairs) -> list[NLIScores]`

`NLIScores` contains:
- `entailment`
- `neutral`
- `contradiction`

These probabilities sum to `1.0`.

Implementations:
- `_RealNLIModel`: HuggingFace-backed, using `cross-encoder/nli-deberta-v3-base`
- `_FinetunedNLIModel`: same interface, but loads a local fine-tuned checkpoint

The model is lazy-loaded so importing the module does not trigger a download.

### Coherence Checking
**File:** `diagnostics/coherence.py`

#### `annotate_weights(g, model)`
Scores each argumentative edge using the NLI model.

For each edge `(src_text, tgt_text)`:
- cache `_nli_scores`
- derive `weight` from the edge label
  - support -> entailment score
  - attack -> contradiction score

This is the most computationally expensive stage in the pipeline.

#### `check_coherence(g, model, min_severity)`
Consumes cached NLI scores and flags semantic label mismatches:

- support edge with `contradiction > entailment` -> `SUPPORT_CONTRADICTED`
- attack edge with `entailment > contradiction` -> `ATTACK_ENTAILED`

Each failure is represented as:

```python
CoherenceFailure(
    edge_id,
    src_id,
    tgt_id,
    edge_label,
    nli_scores,
    failure_code,
    severity,
)
```

`min_severity` suppresses low-confidence noise.

### Witnesses
**File:** `diagnostics/witnesses.py`

The witness layer converts failures into repair suggestions.

#### `classify_failures(failures)`
Rules:
- `SUPPORT_CONTRADICTED` with contradiction >= 0.5 -> `FLIP_LABEL`
- `SUPPORT_CONTRADICTED` with contradiction < 0.5 -> `DELETE_EDGE`
- `ATTACK_ENTAILED` with entailment >= 0.5 -> `FLIP_LABEL`
- `ATTACK_ENTAILED` with entailment < 0.5 -> `DELETE_EDGE`

This produces `Witness(failure, repair_hint)` objects without any new model calls.

#### `find_unsupported_claims(g, model)`
Scans incoming support edges for each node and checks whether any of them actually carry meaningful entailment. A node may appear structurally supported while remaining semantically unsupported.

### Repair Operations
**File:** `repair/ops.py`

All repair functions operate on **deep copies** of the graph.

#### `flip_label(g, edge_id)`
- locates edge by its `id`
- copies the graph
- flips `support <-> attack`
- re-derives `weight` from cached NLI scores

#### `delete_edge(g, edge_id)`
- locates edge by `id`
- copies the graph
- removes the edge

#### `apply_repairs(g, witnesses)`
- applies witnesses sequentially to a working copy
- gracefully skips operations on edges already removed by earlier repairs
- returns:

```python
RepairResult(graph, applied, skipped)
```

This preserves a complete audit log for explanation and debugging.

---

## Fallacy Detection

**File:** `diagnostics/fallacies.py`

DIANOIA supports two fallacy-detection tiers.

### Tier 1 - Deterministic Graph Checks

#### Circular reasoning
`_detect_circular_reasoning(g)` finds cycles where **every** edge is labeled `support`.

Pattern:
- A supports B
- B supports C
- C supports A

This is treated as circular support, not merely a generic cycle.

#### Missing support
`_detect_missing_support(g)` flags nodes that make outgoing claims about other arguments but have no incoming support edges of their own.

These checks are deterministic and return `FallacyResult` objects with:
- `tier = 1`
- `score = 1.0`

### Tier 2 - Learned Fallacy Classification

`_LogicFallacyModel` wraps the HuggingFace model `minjingbo/logic`.

Supported labels include:
- `ad_hominem`
- `appeal_to_authority`
- `appeal_to_emotion`
- `false_causality`
- `false_dilemma`
- `hasty_generalization`
- `slippery_slope`
- `red_herring`
- `circular_reasoning`
- `ad_populum`
- `irrelevant_conclusion`
- `intentional_fallacy`
- `fallacy_of_credibility` (mapped to `appeal_to_authority`)

`_detect_with_model(g, model, min_score)` batches node texts and returns model-based `FallacyResult` objects above the configured threshold.

### Top-level API
`detect_fallacies(g, model, min_score, run_tier1, run_tier2)` composes both tiers.

---

## Argumentation Semantics

**File:** `graph/semantics.py`

This layer implements Dung-style abstract argumentation along with graded strength metrics.

### Helper functions
- `_attackers(g, node)`
- `_is_conflict_free(g, s)`
- `_defends(g, s, node)`
- `_is_admissible(g, s)`

### `grounded_extension(g)`
Computes the unique grounded extension by iterating the characteristic function to a least fixed point.

Interpretation: these are the most sceptically acceptable arguments.

### `preferred_extensions(g)`
Finds all maximal admissible sets via backtracking with pruning.

Interpretation: these are the major admissible "possible worlds" of the framework.

### `stable_extensions(g)`
Filters preferred extensions to those that attack every node outside the set.

Stable extensions need not exist in every argumentation framework.

### `grounded_degree(g)`
Implements the continuous h-categorizer of Besnard and Hunter (2001):

```text
strength(a) = 1 / (1 + Σ weight(b->a) * strength(b))
```

where the sum ranges over attackers of `a`.

### `bipolar_degree(g)`
Extends the score to include both support and attack:

```text
strength(a) = (1 + Σ w(s->a) * strength(s)) /
              (1 + Σ w(s->a) * strength(s) + Σ w(b->a) * strength(b))
```

Interpretation:
- strong support pushes a score upward
- strong attack pushes it downward
- isolated nodes default to `1.0`

In practice, `bipolar_degree` is the most useful ranking signal for claims.

---

## Raw-Text Extraction

**File:** `nlp/extract.py`

This layer allows DIANOIA to run directly on raw text without a pre-annotated corpus.

### 1. Segmentation
`_segment(text)` splits the document into sentence-like units using a lightweight regex over `.`, `!`, and `?` boundaries.

Output:
- sentence text
- `(start_char, end_char)` offsets

### 2. ADU Typing
`_type_adus(segments)` performs zero-shot NLI against three hypotheses:

- "This sentence is the main claim..."
- "This sentence is a claim..."
- "This sentence is a premise..."

The type with highest entailment wins.

### 3. Relation Detection
`_detect_relations(adus)` scores all ordered sentence pairs, skipping self-pairs.

If either:
- entailment >= threshold, or
- contradiction >= threshold,

then an edge is emitted.

Label assignment:
- support if entailment dominates
- attack if contradiction dominates

Confidence:
- `max(entailment, contradiction)`

The result is a synthetic `CanonDoc` with `corpus="extracted"`.

---

## Engine Orchestration

**File:** `engine.py`

`ArgumentEngine` is the orchestrator that runs the entire pipeline and returns an `AnalysisResult`.

At a high level, the engine:
1. accepts either raw text or a structured canonical document
2. constructs the graph
3. runs structural checks
4. annotates semantic edge weights via NLI
5. diagnoses coherence failures
6. generates witness-based repair hints
7. applies repairs on a copied graph
8. detects fallacies
9. computes argumentation semantics before and after repair
10. packages everything into an `AnalysisResult`

The engine caches NLI scores on edges so later stages can reuse them without re-scoring the same pairs.

---

## Explanations and Evaluation

### Explanations
**Directory:** `explain/`

Three pure functions convert `AnalysisResult` into user-facing outputs.

#### `bullet_points(result)`
Produces one sentence per finding, grouped across:
- sanity warnings
- coherence failures
- repair suggestions
- fallacies
- unsupported claims

#### `narrative(result)`
Builds a flowing prose summary with this structure:
- overview
- structural issues
- coherence issues
- repairs
- fallacies
- unsupported claims
- bipolar-degree summary
- grounded-extension summary

#### `structured_report(result)`
Returns a JSON-serializable dictionary with:
- rounded float values
- lists sorted by severity
- embedded narrative text

This is the format most suitable for serving from the API.

### Evaluation
**File:** `eval/metrics.py`

#### `coherence_precision_recall(predicted, gold)`
Computes precision, recall, and F1 for coherence failures.

Matching can be done by:
- `edge_id`
- `failure_code`
- both

#### `repair_minimality(original_graph, repaired_graph)`
Measures how many edge flips and deletions were required.

#### `fallacy_f1(predicted, gold)`
Computes per-type and macro F1 across the fallacy label set.

#### `extraction_f1(predicted_doc, gold_doc)`
Evaluates the extraction layer on three axes:
- ADU boundary F1 via span IoU >= 0.5
- type accuracy on matched ADUs
- edge F1 on `(src, tgt, label)`

---

## API

**Directory:** `api/`

### `app.py`
Creates the FastAPI application and stores a single shared `ArgumentEngine` instance on `app.state.engine`.

That design ensures the NLI model is loaded once and then reused across HTTP requests.

### `schemas.py`
Defines Pydantic schemas for request and response validation, including:
- `TextRequest`
- `CanonDocRequest`
- `AnalysisResultSchema`

### `routes.py`
Exposes three endpoints:

#### `GET /health`
```json
{"status": "ok"}
```

#### `POST /analyze`
Accepts raw text and calls `engine.analyze_text()`.

#### `POST /analyze/structured`
Accepts a pre-labeled structured document, converts it into `CanonDoc`, and calls `engine.analyze()`.

Serialization is handled by `_serialize()`, which maps internal dataclasses to their Pydantic equivalents.

Error handling:
- validation failures -> HTTP 422
- unexpected runtime failures -> HTTP 500

---

## CLI

**File:** `cli/build_data.py`

Registered through `pyproject.toml` as:

```bash
dianoia-build-data --corpus argmicro aae2 cdcp
```

The command dispatches to the corresponding dataset builder and can optionally run a validation/sanity report afterward.

Reported statistics include:
- document count
- average ADUs per document
- average edges per document
- zero-edge document rate
- label distribution

Because the report reloads the generated JSONL through `read_jsonl()`, every emitted document is validated again.

---

## Training

**File:** `scripts/finetune_nli.py`

This script fine-tunes a DeBERTa-v3 classifier on argument-relation pairs derived from the corpora.

### Pair construction
- support edge -> `(src_text, tgt_text, ENTAILMENT)`
- attack edge -> `(src_text, tgt_text, CONTRADICTION)`
- random non-edge pair -> `(src_text, tgt_text, NEUTRAL)`

### Engineering decisions

#### Weighted loss
Uses sqrt inverse-frequency weighting (`power=0.5`) rather than full inverse frequency.

Rationale:
- contradiction is rare
- full inverse-frequency weighting creates unstable gradients
- `power=0.5` improves balance without overwhelming training stability

#### Apple Silicon / MPS fix
Some DeBERTa operations run internally in float16 on MPS even when the model is loaded in float32. The fix is to cast with:

```python
logits.float()
```

before `F.cross_entropy`.

#### Device selection
`--device cpu` is the reliable option because DeBERTa-v3 disentangled attention produced incorrect gradients on MPS.

#### Offline mode
Uses:
- `HF_HUB_OFFLINE=1`
- `local_files_only=True`

when weights are already cached locally.

#### Transformers version compatibility
The script adapts to API changes in `transformers >= 5.0`, including:
- `no_cuda` -> `use_cpu`
- `tokenizer` -> `processing_class`

Version detection is performed dynamically at runtime.

---

## End-to-End Request Flow

### Example request

```json
{
  "text": "We should ban plastics. They harm the ocean."
}
```

### Execution path

```text
api/routes.py
  -> engine.analyze_text()
     -> nlp/extract.py
        - sentence segmentation
        - zero-shot ADU typing
        - NLI-based relation detection
        - synthetic CanonDoc construction
     -> engine._run_pipeline()
        -> graph/build.py
        -> graph/sanity.py
        -> diagnostics/coherence.py::annotate_weights
        -> diagnostics/coherence.py::check_coherence
        -> diagnostics/witnesses.py::classify_failures
        -> diagnostics/witnesses.py::find_unsupported_claims
        -> repair/ops.py
        -> diagnostics/fallacies.py
        -> graph/semantics.py
  -> api/routes.py::_serialize()
  -> HTTP 200 response
```

The key operational detail is that **NLI scores are computed once, cached on edges, and then reused by coherence checking, witness generation, repair, and unsupported-claim detection.**

---

## Design Principles

Several design choices recur throughout the project.

### 1. One canonical representation
Every corpus is normalized into `CanonDoc`, and everything downstream starts from that common schema.

### 2. Fail early
Validation runs at ingestion and serialization boundaries so malformed data is caught immediately.

### 3. Separate structure from semantics
Graph topology and NLI-based semantic agreement are distinct diagnostic layers.

### 4. Never mutate the original graph
Repairs operate on copies and preserve an audit trail of applied and skipped edits.

### 5. Cache expensive model outputs
Edge-level NLI scoring is reused rather than recomputed.

### 6. Keep the system composable
Converters, diagnostics, semantics, repair, explanation, API, and evaluation are cleanly separable.

---

## Repository Sketch

```text
types.py
engine.py

api/
  app.py
  routes.py
  schemas.py

cli/
  build_data.py

datasets/
  argmicro.py
  aae2.py
  cdcp.py

diagnostics/
  coherence.py
  witnesses.py
  fallacies.py

eval/
  metrics.py

explain/
  __init__.py

graph/
  build.py
  sanity.py
  semantics.py

io/
  jsonl.py

nlp/
  nli.py
  extract.py

repair/
  ops.py

scripts/
  finetune_nli.py
```

---

## In One Sentence

DIANOIA is a layered argument-analysis framework that unifies corpus conversion, graph diagnostics, semantic consistency checking, minimal repair, fallacy detection, argumentation semantics, and explanation behind a single canonical representation and a reusable engine.


