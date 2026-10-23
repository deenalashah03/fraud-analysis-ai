# Fraud Analysis AI Platform

An AI-powered fraud analysis and investigation platform that combines machine learning, deterministic risk signals, and explainable AI to analyze suspicious financial transactions.

The platform helps investigators understand **why a transaction appears risky** by combining predictive fraud scoring with supporting evidence.

---

# Overview

Fraud Analysis AI is a portfolio-grade fraud investigation assistant designed to demonstrate modern fraud analysis patterns used in financial systems.

The system focuses on:

* Machine learning-based fraud risk prediction
* Rule-based risk signal generation
* Explainable fraud analysis
* Evidence aggregation
* Investigation recommendations

The platform is designed to evolve toward advanced capabilities such as Retrieval-Augmented Generation (RAG) and agent-assisted investigation workflows.

---

# Problem Statement

Financial institutions process millions of transactions every day. Identifying suspicious activity requires analyzing large volumes of transactional data while maintaining explainability, consistency, and auditability.

Traditional fraud systems typically rely on:

## Rule-Based Systems

Advantages:

* Easy to understand
* Easy to audit
* Effective for known fraud patterns

Limitations:

* Difficult to adapt to new fraud behaviors
* Require continuous rule maintenance

---

## Machine Learning Models

Advantages:

* Detect complex behavioral patterns
* Learn from historical transaction data

Limitations:

* Predictions may lack transparency
* Require explainability for investigation workflows

---

This project explores a hybrid fraud analysis approach combining:

* Machine learning predictions
* Deterministic risk signals
* Explainable investigation outputs

---

# What Is Fraud?

Fraud is unauthorized, deceptive, or malicious activity intended to cause financial loss.

Examples:

* Account takeover
* Card-not-present fraud
* Identity fraud
* Payment fraud
* Transaction abuse
* Synthetic identity fraud

---

# Project Goals

The goal of this platform is to:

* Analyze suspicious transactions
* Estimate fraud risk
* Identify contributing risk factors
* Provide investigation evidence
* Generate explainable fraud assessments
* Recommend investigation actions

The system assists investigators; it does not autonomously approve or block transactions.

---

# Current MVP Architecture

The MVP implements a hybrid fraud analysis pipeline.

```text
                     Transaction Request

                              |
                              v

                 Fraud Analysis Service

                              |
              --------------------------------
              |                              |
              v                              v

       ML Fraud Analysis              Rule Engine
              |
              |
        -----------------
        |               |
        v               v

   Fraud Score      SHAP Explanation


              |
              |
              --------------------------------
                             |
                             v

        Fraud Assessment & Recommendation Layer

                             |
                             v

                 Investigation Summary
```

---

# Core Components

## 1. Machine Learning Fraud Analysis

The ML model analyzes transaction characteristics and predicts fraud likelihood.

Responsibilities:

* Learn patterns from historical transaction data
* Generate fraud probability scores
* Identify suspicious behavioral patterns

Example output:

```json
{
  "fraud_probability": 0.82,
  "risk_level": "HIGH"
}
```

Potential models:

* XGBoost
* Logistic Regression
* Random Forest
* Autoencoder (for anomaly detection)

---

## 2. SHAP Explainability Layer

SHAP provides transparency into ML model predictions.

The purpose is to explain:

* Which features influenced the fraud score
* Why the model considers a transaction risky

Example:

```json
{
  "fraud_probability": 0.82,
  "top_factors": [
    "NEW_DEVICE",
    "HIGH_AMOUNT",
    "LOCATION_MISMATCH"
  ]
}
```

SHAP does not make fraud decisions. It provides explainability evidence for investigators.

---

## 3. Rule Engine

The rule engine generates deterministic risk signals.

Rules provide additional evidence but do not make the final fraud decision.

Examples:

* High transaction amount
* New device activity
* Location mismatch
* Unusual transaction frequency
* Blacklisted entities

Example output:

```json
{
  "risk_signals": [
    "HIGH_AMOUNT",
    "NEW_DEVICE"
  ]
}
```

---

## 4. Fraud Assessment & Recommendation Layer

This layer combines:

* ML fraud probability
* SHAP explanation
* Rule-based risk signals

to produce an investigation assessment.

Example:

```json
{
  "risk_score": 0.82,
  "risk_level": "HIGH",
  "risk_signals": [
    "NEW_DEVICE",
    "LOCATION_MISMATCH"
  ],
  "recommendation": "MANUAL_REVIEW"
}
```

---

## 5. Investigation Summary

The platform generates a human-readable investigation summary.

Example:

```
Transaction shows elevated fraud risk.

Evidence:
- Transaction amount significantly exceeds normal behavior
- New device detected
- Location differs from historical activity

Recommendation:
Perform additional verification.
```

---

# Future Architecture Extensions

The MVP architecture is intentionally designed to support additional capabilities without changing the core fraud analysis pipeline.

---

# Retrieval-Augmented Generation (RAG)

Future versions can add a knowledge retrieval layer.

Purpose:

* Retrieve similar historical fraud cases
* Provide investigation context
* Improve investigation explanations

Potential knowledge sources:

* Historical fraud investigations
* Fraud case summaries
* Security guidelines
* Investigation notes

Future flow:

```text
                 Fraud Assessment

                         |
                         v

              Vector Database Retrieval

                         |
                         v

              Similar Fraud Cases

                         |
                         v

           Enhanced Investigation Summary
```

---

# Investigation Agent Workflow

Future versions can introduce a LangGraph-based Investigation Agent.

The agent does not replace fraud models.

Its purpose is to coordinate investigation steps by combining:

* ML fraud scoring
* Rule evaluation
* Historical case retrieval

Future architecture:

```text
                 Transaction Request

                         |
                         v

              Investigation Agent

                         |
          --------------------------------
          |              |               |
          v              v               v

      ML Tool       Rule Tool        RAG Tool

                         |
                         v

             Investigation Result

                         |
                         v

              Recommended Action
```

---

# Memory Architecture (Future)

## Short-Term Memory

Maintains context during a single investigation.

Stores:

* Transaction details
* Fraud score
* Risk signals
* Retrieved evidence
* Investigation findings

Purpose:

* Maintain investigation context across workflow steps

---

## Long-Term Memory

Stores knowledge across investigations.

Potential storage:

* Vector database

Stores:

* Historical fraud cases
* Fraud patterns
* Investigation outcomes

Purpose:

* Retrieve similar scenarios
* Improve investigation quality

---

# Technology Stack

## Backend

* Java
* Spring Boot

## AI Service

* Python
* FastAPI

## Machine Learning

* XGBoost
* Scikit-learn
* Pandas
* NumPy

## Explainability

* SHAP

## Future GenAI Components

* LangChain
* LangGraph
* Vector Database
* LLM APIs

## Infrastructure

* Docker
* AWS

---

# Repository Structure

```text
fraud-analysis-ai/

├── README.md
│
├── docs/
│   ├── architecture.md
│   └── design-decisions.md
│
├── diagrams/
│
├── fraud-engine-python/
│
│   ├── training/
│   │   ├── analyze_dataset.py
│   │   ├── preprocess.py
│   │   ├── train_model.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   └── fraud_model.pkl
│   │
│   ├── rules/
│   │   └── rule_engine.py
│   │
│   └── app.py
│
├── backend-java/
│
├── data/
│
└── requirements.md
```

---

# Running Locally

## Clone Repository

```bash
git clone <repository-url>

cd fraud-analysis-ai
```

---

## Setup Python Environment

```bash
cd fraud-engine-python

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Start Fraud Analysis API

```bash
uvicorn app:app --reload
```

---

# Design Principles

This project follows these principles:

* Build a working MVP before adding advanced AI capabilities
* Separate prediction from decision-making
* Keep fraud analysis explainable
* Combine machine learning with deterministic controls
* Use GenAI for reasoning and context, not fraud detection
* Design toward enterprise-style extensibility

---

# Current Project Focus

Current implementation:

✅ Fraud risk prediction
✅ Rule-based risk signals
✅ SHAP-based explainability
✅ Fraud assessment generation
✅ Investigation recommendations

Future extensions:

* RAG-based fraud knowledge retrieval
* LangGraph Investigation Agent
* Stateful investigation workflows
* Enterprise integrations

---

# Project Vision

The long-term vision is to create an AI-assisted fraud investigation platform that helps analysts make faster, better-informed decisions by combining predictive models, business rules, explainable AI, and intelligent reasoning.
