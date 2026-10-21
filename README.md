# Detecting Hallucinations in Large Language Model Outputs: A Feature-Based Classification Approach

**A machine learning study on detecting factual inconsistencies in Large Language Model (LLM) generated text.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Focus](https://img.shields.io/badge/Focus-LLM%20Reliability-purple)](.)

## Overview

Large Language Models (LLMs) can generate fluent, convincing responses that contain inaccurate, unsupported, or fabricated information. These hallucinations present a major challenge to the reliability of AI-generated content, particularly in applications where factual correctness matters.

This project investigates the identification of hallucinated information in LLM outputs using machine learning and systematic evaluation. Using the **HaluEval benchmark**, the study explores hallucination detection, model performance, and potential dataset-related biases that may influence evaluation results.

Rather than treating a high classification score as sufficient evidence of a reliable detector, this work also considers how dataset characteristics can affect the interpretation of model performance.

## Research Objectives

- Develop and evaluate machine learning approaches for identifying hallucinated information in LLM-generated text.
- Investigate classification performance using appropriate evaluation metrics.
- Examine whether dataset characteristics, particularly text length, introduce biases into hallucination detection.
- Compare experimental results across relevant subsets of the benchmark.
- Identify limitations that should be considered when evaluating hallucination detection systems.

## Dataset

The project uses the **HaluEval benchmark**, a dataset designed to support research on hallucinations in LLM-generated responses.

The experiments include a summarization-focused subset and investigate the relationship between text characteristics and hallucination classification.

| Attribute | Description |
|---|---|
| Dataset | HaluEval |
| Research task | Hallucination detection |
| Data modality | Text |
| Learning task | Binary classification, subject to the dataset's label definition |
| Evaluation | F1-score, ROC-AUC, and other metrics used in the experiments |

**Data note:** Refer to the original dataset documentation and the project notebooks for the exact subset, sample counts, label distribution, preprocessing steps, and train/test split.

## Methodology

The project follows a structured experimental workflow:

1. **Data preparation** — Load the benchmark data and prepare the text and labels for experimentation.
2. **Exploratory data analysis** — Examine dataset distributions and relevant text characteristics.
3. **Preprocessing and feature preparation** — Apply the transformations required by the implemented detection approaches.
4. **Model development** — Train and evaluate the implemented machine learning models.
5. **Performance evaluation** — Measure classification performance using appropriate metrics, including F1-score and ROC-AUC where applicable.
6. **Bias investigation** — Examine whether text length or other dataset characteristics may influence the detector's predictions.
7. **Results analysis** — Interpret model performance alongside potential limitations and sources of bias.

The notebooks contain the implementation details and experimental workflow. Consult them for the precise algorithms, features, hyperparameters, and evaluation protocol.

## Key Results

### Classifier Performance

The following results were reported for the summarization-focused hallucination detection experiment.

| Model | F1 | Precision | Recall | AUC |
|---|---:|---:|---:|---:|
| **Gradient Boosting** | **0.728** | 0.746 | 0.710 | 0.801 |
| Random Forest | 0.726 | 0.738 | 0.714 | 0.794 |
| Logistic Regression | 0.723 | 0.752 | 0.696 | 0.805 |
| SVM | 0.722 | 0.759 | 0.688 | 0.806 |
| KNN | 0.721 | 0.723 | 0.720 | 0.784 |
| Decision Tree | 0.691 | 0.708 | 0.674 | 0.758 |

Gradient Boosting achieved the highest F1-score (0.728) in this comparison. SVM achieved the highest AUC (0.806), closely followed by Logistic Regression (0.805). The best model therefore depends on whether the priority is balancing precision and recall at the selected threshold or ranking examples across thresholds.

*Confirm the evaluation split, positive-class definition, and metric calculation in the notebooks before treating these figures as independently reproducible results.*

### Datasets

| Dataset | Source | Purpose |
|---|---|---|
| HaluEval (summarization) | [Hugging Face](https://huggingface.co/datasets/pminervini/HaluEval) | Training and evaluating classifiers |
| HaluEval (QA) | [Hugging Face](https://huggingface.co/datasets/pminervini/HaluEval) | Investigating potential length bias |
| TruthfulQA | [Hugging Face](https://huggingface.co/datasets/truthful_qa) | Generating LLM responses for experimentation |

Dataset versions, selected configurations, and the exact split used for each experiment should be documented for reproducibility.

### NLP Features

Eight NLP features were computed per response against its source document.

| Feature | Description |
|---|---|
| `response_len` | Word count of the response |
| `hedge_count` | Count of uncertainty markers (e.g., “might,” “possibly”) |
| `rouge1` | Unigram overlap with the source document (ROUGE-1 F1) |
| `rougeL` | Longest common subsequence overlap (ROUGE-L F1) |
| `bleu` | N-gram precision using smoothed sentence BLEU |
| `sem_sim` | Cosine similarity of sentence embeddings (`all-MiniLM-L6-v2`) |
| `bertscore_f1` | Token-level semantic similarity (`distilbert-base-uncased`) |
| `nli_score` | NLI entailment score (`cross-encoder/nli-deberta-v3-small`) |

These features capture complementary signals, including response length, lexical overlap, semantic similarity, and estimated entailment. They are indicators of consistency rather than definitive proof that a response is factually correct or incorrect.

### LLMs Queried

The following models and providers were used for response generation.

| Model | Provider / route | API or platform |
|---|---|---|
| Mistral Small | Mistral AI | [Mistral Console](https://console.mistral.ai/) |
| Llama 3.3 70B | Meta model via Groq | [Groq Console](https://console.groq.com/) |
| Llama 3.1 8B | Meta model via SambaNova / Hugging Face | [Hugging Face](https://huggingface.co/) |

*Provider access, model availability, and free-tier limits may change. Never commit API keys or other credentials to the repository.*

### Interpretation

- **Evaluation requires context:** Performance depends on the dataset, label distribution, and evaluation protocol.
- **Dataset bias matters:** Text length and other structural characteristics can potentially provide shortcuts for classification rather than genuine evidence of hallucination.
- **Aggregate metrics are not enough:** Error analysis and subgroup evaluation can reveal weaknesses hidden by an overall score.
- **Reliable detection requires further validation:** Performance on a benchmark does not automatically establish generalization to unseen models, domains, or real-world user queries.

## Repository Structure

```text
Identifying-hallucinated-information-in-LLM-outputs/
│
├── py_notebooks/
│   └── Jupyter notebooks for analysis and experiments
│
├── outputs_and_extracted_csv(s)/
│   └── Exported outputs and processed data files
│
├── Concept Paper.pdf
├── Project Report.pdf
├── Requirements.txt
├── LICENSE
└── README.md
```

The descriptions above summarize the apparent roles of the repository files. Check the actual contents of each folder before adding specific notebook names or dataset filenames.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab
- The Python libraries listed in `Requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/dalalaadi30/Identifying-hallucinated-information-in-LLM-outputs.git
cd Identifying-hallucinated-information-in-LLM-outputs
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r Requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open the relevant notebook in `py_notebooks/` and follow its execution order.

### Dataset Setup

If the notebook expects an externally downloaded dataset, obtain it from the original HaluEval source and place the files in the location expected by the notebook.

Review any file paths, environment-specific settings, and preprocessing requirements before running the experiments.

**Reproducibility:** The exact dataset version, random seeds, data splits, model configurations, and notebook execution order should be documented to allow others to reproduce the reported results.

## Evaluation Metrics

**F1-score**

The harmonic mean of precision and recall. It is useful when evaluating a detector that must balance false positives against missed hallucinations.

**ROC-AUC**

Measures how well a model distinguishes between the two classes across different classification thresholds, assuming the binary-classification setup used in the experiments.

Both metrics should be interpreted alongside the confusion matrix, class distribution, and per-class precision and recall.

## Limitations

This project should be interpreted within the limits of its experimental setup:

- Benchmark performance may not generalize to other LLMs, domains, or styles of generated text.
- Dataset-specific patterns can inflate apparent performance if the model learns shortcuts.
- Detecting a potentially hallucinated response is not the same as independently verifying every factual claim against authoritative evidence.
- Results depend on label quality, preprocessing, the selected model, and the evaluation split.
- Additional testing on out-of-distribution examples and independent benchmarks is necessary to establish broader reliability.

## Future Work

Potential extensions include:

- Evaluating generalization across different LLMs and datasets.
- Testing robustness to changes in response length and writing style.
- Performing error analysis on false positives and false negatives.
- Comparing classification-based detection with evidence-grounded factual verification.
- Exploring retrieval-augmented generation (RAG) as a way to supply supporting evidence.
- Investigating whether retrieval quality and evidence relevance can help explain downstream hallucinations.

These are potential extensions rather than completed experiments unless documented in the existing notebooks.

## Research Motivation

This project reflects a broader interest in **reliable and trustworthy AI systems**. As LLMs are increasingly used to summarize information, answer questions, and support decision-making, understanding when their outputs are inaccurate is essential.

A key research direction is to move beyond simply identifying unreliable outputs and investigate *why* they occur, including the role of dataset biases, missing evidence, and retrieval failures.

## Project Materials

- **Source code and notebooks:** [Browse the repository](https://github.com/dalalaadi30/Identifying-hallucinated-information-in-LLM-outputs)
- **Concept paper:** [`Concept Paper.pdf`](Concept%20Paper.pdf)
- **Project report:** [`Project Report.pdf`](Project%20Report.pdf)
- **Dependencies:** [`Requirements.txt`](Requirements.txt)
- **License:** [MIT License](LICENSE)

## Author

**Aaditya Dalal**

Interested in machine learning, natural language processing, retrieval-augmented generation, and the reliability of AI systems.

[GitHub Profile](https://github.com/dalalaadi30)

---

*This repository documents an experimental study of hallucination detection. Reported metrics should be interpreted in the context of their corresponding experiments, datasets, and evaluation protocols.*
