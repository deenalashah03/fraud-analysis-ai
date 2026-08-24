                 ModelOrchestrator
                       │
        ┌──────────────┴──────────────┐
        │                             │
DEVELOPMENT                    INFERENCE
│                             │
▼                             ▼
preprocess                       model_name
│                             │
▼                             ▼
model-specific                 MODEL_REGISTRY
model                           │
│                             ▼
┌────┼─────────────┐        finalized model
│    │             │              │
baseline SHAP      experiments      ▼
│                  │           probability
▼                  ▼               │
XGBoost            CSVs             ▼
threshold
│
▼
prediction