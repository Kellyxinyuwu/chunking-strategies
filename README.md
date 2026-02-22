# Chunking Strategies for RAG — A Comprehensive Tutorial

<img src="./media/chunking.png" width="800">

A hands-on tutorial exploring document chunking strategies for RAG (Retrieval Augmented Generation) systems. Based on [Chroma's research](https://research.trychroma.com/evaluating-chunking), this project walks through five chunking methods, evaluates them, and distills practical recommendations for your own RAG projects.

---

## Table of Contents

- [Why Chunking Matters for RAG](#why-chunking-matters-for-rag)
- [How Chunking Fits in the RAG Pipeline](#how-chunking-fits-in-the-rag-pipeline)
- [Quick Start](#quick-start)
- [Chunking Strategies Covered](#chunking-strategies-covered)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Findings & RAG Recommendations](#key-findings--rag-recommendations)
- [Takeaways for Your RAG Project](#takeaways-for-your-rag-project)
- [Resources](#resources)

---

## Why Chunking Matters for RAG

In RAG, you retrieve relevant text chunks and feed them to an LLM as context. **Chunking** is the step that splits documents into these pieces. The quality of your chunks directly affects:

- **Retrieval quality** — Can the retriever find the right chunks?
- **Context efficiency** — Are you sending too much irrelevant text to the LLM?
- **Answer quality** — Does the model get enough coherent context to answer well?

Most people pick one chunking method and stick with it. Chroma's research shows that **choice matters** — some strategies significantly outperform others, and default settings (e.g., OpenAI's) often underperform.

---

## How Chunking Fits in the RAG Pipeline

```
Documents → Chunking → Embedding → Vector DB
                                    ↓
User Query → Embed Query → Retrieve Chunks → LLM (with context) → Answer
```

Chunking happens **before** embedding. Your chunk boundaries determine:
- What gets embedded as a single vector
- What gets retrieved as a unit
- How much context the LLM receives per chunk

**Bad chunking** = relevant info split across chunks, or too much noise in each chunk → worse retrieval and answers.

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) OpenAI API key for LLM Semantic Chunker; otherwise the notebook uses local embeddings

### Setup

```bash
# Clone and enter the project
cd chunking-strategies

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies (chunking_evaluation repo + langchain, etc.)
pip install git+https://github.com/brandonstarxel/chunking_evaluation.git
pip install langchain-experimental langchain-openai langchain-community python-dotenv
```

### Run the Notebook

```bash
jupyter notebook chunking.ipynb
```

**Without OpenAI:** The notebook uses local HuggingFace embeddings for most chunkers. Only the LLM Semantic Chunker requires an API key; that section will skip gracefully if not set.

**With OpenAI:** Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` to enable all sections.

---

## Chunking Strategies Covered

### 1. Character / Token-Based Chunking

- **Idea:** Split at fixed character or token counts.
- **Pros:** Simple, fast, predictable.
- **Cons:** Can split mid-sentence or mid-word; ignores semantics.
- **Use case:** Baseline, or when speed and simplicity matter more than quality.

### 2. Recursive Character / Token Chunking

- **Idea:** Try separators in order (e.g., `\n\n`, `\n`, `.`, ` `) and split at the largest that keeps chunks under the size limit.
- **Pros:** Respects natural boundaries (paragraphs, sentences); widely available (e.g., LangChain's `RecursiveCharacterTextSplitter`).
- **Cons:** Still rule-based, not semantic.
- **Use case:** **Strong default** — Chroma found it competitive with more complex methods.

### 3. Semantic Chunker (Kamradt / LangChain)

- **Idea:** Use embeddings to find where semantic similarity drops between consecutive segments; split there.
- **Pros:** Splits at topic boundaries.
- **Cons:** Needs embeddings; can be slower.
- **Use case:** When you want semantic coherence without full clustering.

### 4. Cluster Semantic Chunker

- **Idea:** Embed small pieces, build a similarity matrix, use dynamic programming to group pieces into chunks that maximize within-chunk similarity.
- **Pros:** Global optimization; often best recall.
- **Cons:** More compute; more complex to implement.
- **Use case:** When you need maximum retrieval quality and can afford the cost.

### 5. LLM Semantic Chunker

- **Idea:** Ask an LLM to identify semantic boundaries in text windows.
- **Pros:** Highest recall in Chroma's evaluation.
- **Cons:** Requires API calls; slower and more expensive.
- **Use case:** When quality is paramount and cost is acceptable.

---

## Evaluation Metrics

Chroma evaluates chunking at the **token level** (not document level), which suits RAG:

| Metric | Formula | What It Measures |
|--------|---------|-------------------|
| **Recall** | \|relevant ∩ retrieved\| / \|relevant\| | % of relevant tokens retrieved |
| **Precision** | \|relevant ∩ retrieved\| / \|retrieved\| | % of retrieved tokens that are relevant |
| **Precision Ω** | Theoretical max precision | Inherent token efficiency of the chunking |
| **IoU** | \|relevant ∩ retrieved\| / (\|relevant\| + \|retrieved\| - \|relevant ∩ retrieved\|) | Overlap quality; penalizes both missing and redundant tokens |

**Higher is better** for all. IoU is especially useful because it balances recall and precision.

---

## Key Findings & RAG Recommendations

### Best Overall

- **ClusterSemanticChunker** (400 tokens): 91.3% recall, strong efficiency.
- **LLMSemanticChunker**: 91.9% recall (highest), but needs API and cost.

### Practical Default

- **RecursiveCharacterTextSplitter** (200–400 tokens, **no overlap**): 88.1% recall, 7.0% precision, 29.9% PrecisionΩ.
- Simple, widely available, and competitive with semantic methods.

### Important Findings

- **Smaller chunks (200–400 tokens)** generally outperform larger ones (800 tokens).
- **Overlap often hurts** — it reduces IoU and precision while only marginally improving recall.
- **OpenAI defaults** (800 tokens, 400 overlap) underperform; avoid blindly copying them.
- **Simple methods can match complex ones** — RecursiveCharacterTextSplitter is a strong baseline.

### If Building Your RAG Today

1. **Start simple:** RecursiveCharacterTextSplitter, 200–400 tokens, no overlap.
2. **If you need more:** ClusterSemanticChunker with 200–400 tokens.
3. **Avoid:** Large chunks (800+ tokens) and heavy overlap unless you have evidence they help.

---

## Takeaways for Your RAG Project

1. **Chunk size is the main lever.** 200–400 tokens is a good default; tune based on your use case.
2. **Overlap is often overrated.** Test with and without; many setups do better without it.
3. **Token-level > character-level.** LLMs work in tokens; use token-based chunking when possible.
4. **Evaluate on your own data.** Chroma’s results are on their corpus; your domain may differ.
5. **Start simple, then optimize.** RecursiveCharacterTextSplitter is a solid first choice; move to semantic chunkers only if you need the gains.
6. **Match embedding model to chunker.** If using semantic chunkers, use the same (or compatible) embedding model for chunking and retrieval.

---

## Resources

- **[Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)** — Chroma's full research and metrics
- **[chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation)** — Open-source implementation used in this repo
- **[Chroma](https://trychroma.com)** — Vector database and embedding tools
- **[Pride and Prejudice](https://www.gutenberg.org/ebooks/1342)** — Sample document (Project Gutenberg)

---

## Project Structure

```
chunking-strategies/
├── chunking.ipynb      # Main tutorial notebook
├── pride_and_prejudice.txt   # Sample document
├── media/              # Images and diagrams
├── .env.example        # Template for API keys (copy to .env)
└── README.md
```
