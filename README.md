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
