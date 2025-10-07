import os
import sys
import mlflow
from urllib.parse import urlparse
import dagshub

from rockfallsecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from rockfallsecurity.entity.config_entity import ModelTrainerConfig
from rockfallsecurity.exception.exception import RockfallSafetyException
from rockfallsecurity.logging.logger import logging

from rockfallsecurity.utils.ml_utils.model.estimator import NetworkModel
from rockfallsecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from rockfallsecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import xgboost as xgb
import numpy as np

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise RockfallSafetyException(e, sys)

    def track_mlflow(self, best_model, classification_metric, params: dict = None, X_example=None, extra_metrics: dict = None, artifacts: dict = None):
        # TrackingURI is set globally in main.py
        with mlflow.start_run():
            # Params
            if params:
                mlflow.log_params(params)
            # Metrics
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision", classification_metric.precision_score)
            mlflow.log_metric("recall_score", classification_metric.recall_score)
            if extra_metrics:
                for k, v in extra_metrics.items():
                    try:
                        mlflow.log_metric(k, float(v))
                    except Exception:
                        pass
            # Signature
            try:
                from mlflow.models.signature import infer_signature
                signature = infer_signature(X_example, best_model.predict(X_example) if X_example is not None else None)
            except Exception:
                signature = None
            mlflow.sklearn.log_model(best_model, artifact_path="model", signature=signature, input_example=X_example)
            # Optional text artifacts (e.g., reports, confusion matrices)
            if artifacts:
                for name, content in artifacts.items():
                    try:
                        mlflow.log_text(str(content), f"{name}.txt")
                    except Exception:
                        pass
            

    def train_model(self, X_train, y_train, X_test, y_test):
        # Check class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution in y_train:", dict(zip(unique, counts)))
        logging.info(f"Class distribution in y_train: {dict(zip(unique, counts))}")

        # Check for data leakage (overlap between train and test)
        overlap = np.intersect1d(
            X_train.view([('', X_train.dtype)] * X_train.shape[1]),
            X_test.view([('', X_test.dtype)] * X_test.shape[1])
        )
        print(f"Number of overlapping rows between train and test: {len(overlap)}")
        logging.info(f"Number of overlapping rows between train and test: {len(overlap)}")

        # Feature selection to reduce overfitting
        from sklearn.feature_selection import SelectFromModel
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=32, random_state=42, class_weight="balanced"),
            threshold="median",
        )
        selector.fit(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)
        print(f"Selected {X_train_sel.shape[1]} features out of {X_train.shape[1]}")

        # IsolationForest anomaly detection with threshold tuning
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import (
            classification_report,
            average_precision_score,
            precision_recall_curve,
            confusion_matrix,
        )

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        total = max(n_pos + n_neg, 1)
        # Slightly inflate contamination to be more recall-friendly, with a safe floor
        prior = (n_pos / total) if total > 0 else 0.0
        contamination = max(min(prior * 2.0 if prior > 0 else 0.0 + 1e-4, 0.5), 1e-4)

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X_train_sel)

        # decision_function: larger => more normal, smaller => more anomalous
        # invert so that higher score means more likely anomaly (positive class)
        scores_train = -iso.decision_function(X_train_sel)
        scores_test = -iso.decision_function(X_test_sel)

        # Tune threshold on train to maximize F1 for positive class
        precisions, recalls, thresholds = precision_recall_curve(y_train, scores_train)
        # Choose threshold by maximizing both F1 and F2 (recall-heavy), then pick best
        tuned_threshold = None
        if thresholds is not None and len(thresholds) > 0:
            p = precisions[1:]
            r = recalls[1:]
            # F1
            f1s = 2 * (p * r) / np.maximum(p + r, 1e-12)
            # F2 (beta=2)
            beta2 = 4.0
            f2s = (1 + beta2) * (p * r) / np.maximum(beta2 * p + r, 1e-12)
            # Select best by F2 first, then fallback to F1
            idx_f2 = int(np.nanargmax(f2s)) if np.isfinite(f2s).any() else None
            idx_f1 = int(np.nanargmax(f1s)) if np.isfinite(f1s).any() else None
            if idx_f2 is not None and np.isfinite(f2s[idx_f2]):
                tuned_threshold = float(thresholds[idx_f2])
            elif idx_f1 is not None and np.isfinite(f1s[idx_f1]):
                tuned_threshold = float(thresholds[idx_f1])
        if tuned_threshold is None:
            # fallback: pick 99th percentile of scores to flag some anomalies
            tuned_threshold = float(np.percentile(scores_train, 99.0))

        # Evaluate with tuned threshold
        anomaly_pred_bin_test = (scores_test >= tuned_threshold).astype(int)
        print("\nIsolationForest anomaly detection report (threshold-tuned):")
        report_test = classification_report(y_test, anomaly_pred_bin_test, zero_division=1)
        print(report_test)
        ap_test = average_precision_score(y_test, scores_test)
        print(f"Average Precision (PR AUC): {ap_test:.4f}")
        # Confusion matrix for test
        cm_test = confusion_matrix(y_test, anomaly_pred_bin_test).tolist()

        best_model = iso
        best_model_name = "IsolationForest"

        # Model evaluation and logging using tuned threshold
        anomaly_pred_bin_train = (scores_train >= tuned_threshold).astype(int)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=anomaly_pred_bin_train)
        report_train = classification_report(y_train, anomaly_pred_bin_train, zero_division=1)
        cm_train = confusion_matrix(y_train, anomaly_pred_bin_train).tolist()
        self.track_mlflow(
            best_model,
            classification_train_metric,
            params={
                "model": best_model_name,
                "contamination": float(contamination),
                "tuned_threshold": float(tuned_threshold),
                "n_features_selected": int(X_train_sel.shape[1]),
            },
            X_example=X_train_sel[:5],
            extra_metrics={"avg_precision_train": float(average_precision_score(y_train, scores_train))},
            artifacts={
                "classification_report_train": report_train,
                "confusion_matrix_train": cm_train,
            },
        )

        classification_test_metric = get_classification_score(y_true=y_test, y_pred=anomaly_pred_bin_test)
        self.track_mlflow(
            best_model,
            classification_test_metric,
            X_example=X_test_sel[:5],
            extra_metrics={"avg_precision_test": float(ap_test)},
            artifacts={
                "classification_report_test": report_test,
                "confusion_matrix_test": cm_test,
            },
        )

        # Persist final model (wrapper)
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        rockfall_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=rockfall_model)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise RockfallSafetyException(e, sys)