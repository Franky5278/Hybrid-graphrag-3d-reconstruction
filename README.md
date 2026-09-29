# Hybrid-graphrag-3d-reconstruction
Hybrid GraphRAG system for grounded question answering over 3D reconstruction literature using vector retrieval, Neo4j knowledge graphs and LLM synthesis.

# Hybrid GraphRAG for 3D Reconstruction Literature

## Overview

This project implements a Hybrid GraphRAG research assistant for technical literature in unsigned distance fields (UDFs) and 3D surface reconstruction.

The system combines vector retrieval, semantic knowledge graphs and grounded LLM synthesis to support cross-paper technical queries with provenance-aware evidence.

## Dataset

- 6 research papers
- 530 provenance-aware text chunks
- 384-dimensional SentenceTransformer embeddings
- 94 canonical entities
- 143 semantic relations

## Pipeline

PDF Documents
→ Token-aware Chunking
→ SentenceTransformer Embeddings
→ Neo4j Vector Index
→ Entity / Relation Extraction
→ Semantic Knowledge Graph
→ Hybrid Retrieval
→ Grounded LLM Synthesis

## Retrieval

The retrieval pipeline combines:

- Vector similarity search
- Lexical entity detection
- Vector-to-graph bridging
- Multi-hop graph traversal
- Evidence-linked provenance retrieval

## Knowledge Graph

The semantic graph contains entities such as:

- Methods
- Concepts
- Datasets
- Metrics
- Research tasks

Entity normalization and schema validation are used to reduce duplicate or inconsistent graph nodes.

## Grounded Generation

Gemini is used for grounded answer synthesis based on retrieved vector and graph evidence.

Answers preserve:

- Paper-level provenance
- Page-level evidence
- Retrieved source chunks
- Semantic graph relations

## Interface

A Streamlit interface exposes:

- Generated answers
- Retrieved chunks
- Graph seed entities
- Semantic relations
- Provenance evidence
- Retrieval diagnostics

## Example Methods Covered

- VAD
- GeoUDF
- DM-UDF
- SuperUDF
- MPF
- DCUDF2

## Repository Structure

```text
hybrid-graphrag-3d-reconstruction/
├── README.md
├── src/
├── data/
├── configs/
├── screenshots/
├── results/
└── requirements.txt
