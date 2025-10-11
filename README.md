# Rockfall Predictor

Predict rockfall events from sensor and weather data using a clean, end‑to‑end ML pipeline (ingestion → validation → transformation → training) with MongoDB, pandas, scikit‑learn, XGBoost, and MLflow.

## Why this project

Rockfalls are rare but high‑impact. This repo shows how to build a practical classifier for rare events with:

- A resilient data pipeline (works even with sparse positives and time series)
- Sensible validation and feature engineering for sensor signals
- Transparent artifacts, logs, and experiment tracking

## What you get

- Data ingestion from MongoDB to CSV “feature store”
- Time‑aware train/test split (fallback to stratified)
- Feature engineering and preprocessing pipelines
- Model training with calibration/thresholding options
- Metrics and runs tracked in MLflow (local file store under `mlruns/`)

## Project layout (important bits)

```text
RockFall-Predictor/
├─ main.py                       # Orchestrates the full pipeline
├─ requirements.txt              # Python dependencies
├─ data_schema/schema.yaml       # Expected columns and types
├─ rockfallsecurity/
│  ├─ components/
│  │  ├─ data_ingestion.py      # Load from MongoDB → CSV, split train/test
│  │  ├─ data_validation.py     # Schema/column count + drift checks
│  │  ├─ data_transformation.py # Feature engineering + preprocessing
│  │  └─ model_trainer.py       # Train/evaluate models, track with MLflow
│  ├─ entity/                    # Configs and artifact dataclasses
│  ├─ logging/                   # Logging setup
│  └─ constant/training_pipeline/__init__.py  # Global constants
├─ Artifacts/                    # All pipeline outputs per run (auto‑created)
├─ logs/                         # Logs per run (auto‑created)
├─ mlruns/                       # MLflow runs (auto‑created)
├─ push_data.py                  # Load CSV → MongoDB helper
├─ synthetic_data_generator.py   # Build a balanced synthetic dataset
└─ test_mongodb.py               # Quick connectivity check
```

## Prerequisites

- Python 3.10+
- MongoDB Atlas or a reachable MongoDB instance
- Windows PowerShell (commands below use `pwsh`), macOS/Linux work too

## Quick start

1. Clone and enter the folder

```pwsh
git clone https://github.com/DurgeshRajput11/Rockfall-predictor.git
cd Rockfall-predictor
```

1. Create and fill a .env file

```pwsh
New-Item -ItemType File -Path .env -Force | Out-Null
Add-Content .env "MONGO_DB_URL=your_mongodb_connection_string"
```

1. Install dependencies

```pwsh
pip install -r requirements.txt
```

1. (Optional) Push example data into MongoDB

Place a CSV at `Rockfall_data/final_balanced_dataset.csv` or generate one:

```pwsh
python synthetic_data_generator.py
```

Then push to MongoDB (uses `MONGO_DB_URL`):

```pwsh
python push_data.py
```

1. Run the full pipeline

```pwsh
python main.py
```

This will:

- Read from MongoDB collection `Rockfall_data` in DB `Durgesh_Rajput`
- Create an artifacts folder under `Artifacts/<timestamp>/...`
- Save a feature store CSV, train/test CSVs, transformed arrays, and preprocessing object
- Train models and log metrics/runs under `mlruns/`

Tip: In VS Code, you can also use the Task “Run pipeline”.

## Data and schema

The expected columns are defined in `data_schema/schema.yaml`. Examples:

- Identifiers/categoricals: `location_id`, `seismic_zone`, `timestamp`
- Numericals: pore pressure, precipitation, displacement, accelerations, water table, etc.
- Target: `rockfall_event` (0/1)

Ingestion cleans MongoDB’s `_id` and converts "na" strings to missing values. Train/test split prefers time order if `timestamp` is valid; otherwise it uses a stratified random split (when both classes exist).

## How the pipeline works

1. Ingestion (`data_ingestion.py`)

- Connects to MongoDB with TLS validation via `certifi`
- Exports collection → pandas → CSV feature store
- Splits train/test (time‑aware when possible; ensures positives in test)

1. Validation (`data_validation.py`)

- Checks column count against schema
- Computes drift report with KS‑test on numeric columns

1. Transformation (`data_transformation.py`)

- Engineers rolling features per `location_id` (means/std/max/lag/z‑scores)
- Applies noise filters (moving average, optional bandpass)
- Builds a `ColumnTransformer` with imputers, encoders, and scalers

1. Training (`model_trainer.py`)

- Selects informative features (RandomForest SelectFromModel)
- Trains supervised models (includes XGBoost) and handles class imbalance
- Optionally calibrates/thresholds probabilities
- Logs metrics and artifacts with MLflow

## Configuration and constants

Global constants live in `rockfallsecurity/constant/training_pipeline/__init__.py` (pipeline/paths, file names, split ratios, target column, etc.). Config classes in `rockfallsecurity/entity/config_entity.py` compose these into concrete paths under `Artifacts/<timestamp>/`.

Environment variable required:

- `MONGO_DB_URL` → MongoDB connection string

## Docker (skeleton)

The `Dockerfile` is currently a commented scaffold. To dockerize, you can start with:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENV MONGO_DB_URL=""
CMD ["python", "main.py"]
```

## Troubleshooting

- Cannot connect to MongoDB
   - Verify `MONGO_DB_URL` in `.env`
   - For Atlas, allow your IP and ensure the user has read/write

- No data in train/test
   - Ensure the MongoDB collection has documents and matches the schema
   - Use `synthetic_data_generator.py` + `push_data.py` to create sample data

- MLflow UI
   - Runs are stored under `mlruns/` locally. You can start a UI with:

      ```pwsh
      mlflow ui --backend-store-uri .\mlruns
      ```

## License

MIT

## Acknowledgements

Built with scikit‑learn, XGBoost, pandas, numpy, MLflow, and MongoDB.
