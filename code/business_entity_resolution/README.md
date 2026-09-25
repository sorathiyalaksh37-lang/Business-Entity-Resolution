# Business Entity Resolution — Pipeline

## How to Run (End-to-End)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run from student_resource/ directory
cd /path/to/student_resource

# Phase 1 — EDA
python3 code/business_entity_resolution/src/01_eda.py

# Phase 2 — Blocking
python3 code/business_entity_resolution/src/02_blocking.py

# Phase 3 — Feature Engineering
python3 code/business_entity_resolution/src/03_feature_engineering.py

# Phase 4 — Train Model
python3 code/business_entity_resolution/src/04_train_model.py

# Phase 5 — Predict + Generate Output
python3 code/business_entity_resolution/src/05_predict.py

# Validate output
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Entry Points
- `src/01_eda.py` — EDA report
- `src/05_predict.py` — generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`

## Key Design Decisions
- **Blocking:** Multi-key TF-IDF + phonetic + address-token blocking
- **Features:** rapidfuzz string similarity + sentence-transformer embeddings
- **Model:** LightGBM classifier
- **Threshold:** Optimized for F_0.5 on validation split
