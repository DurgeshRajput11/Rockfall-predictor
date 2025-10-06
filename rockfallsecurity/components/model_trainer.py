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

    def track_mlflow(self, best_model, classification_metric):
    #TrackingURI is setglobally in main.py
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision", classification_metric.precision_score)
            mlflow.log_metric("recall_score", classification_metric.recall_score)
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            

    def train_model(self, X_train, y_train, X_test, y_test):
        # Check class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution in y_train:", dict(zip(unique, counts)))
        logging.info(f"Class distribution in y_train: {dict(zip(unique, counts))}")

        # Check for data leakage (overlap between train and test)
        overlap = np.intersect1d(X_train.view([('', X_train.dtype)]*X_train.shape[1]), X_test.view([('', X_test.dtype)]*X_test.shape[1]))
        print(f"Number of overlapping rows between train and test: {len(overlap)}")
        logging.info(f"Number of overlapping rows between train and test: {len(overlap)}")

        # Feature selection to reduce overfitting
        from sklearn.feature_selection import SelectFromModel
        selector = SelectFromModel(RandomForestClassifier(n_estimators=32, random_state=42, class_weight="balanced"), threshold="median")
        selector.fit(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)
        print(f"Selected {X_train_sel.shape[1]} features out of {X_train.shape[1]}")

        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=32, max_depth=5, random_state=42, n_jobs=-1, class_weight="balanced"),
            "XGBoost": xgb.XGBClassifier(eval_metric='logloss', verbosity=1, n_estimators=32, max_depth=5, random_state=42, scale_pos_weight=scale_pos_weight)
        }
        params = {
            "Random Forest": {'n_estimators': [16, 32], 'max_depth': [3, 5]},
            "XGBoost": {'n_estimators': [16, 32], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]}
        }
        # Add early stopping for XGBoost
        fit_params = {}
        if "XGBoost" in models:
            fit_params["XGBoost"] = {
                "early_stopping_rounds": 10,
                "eval_set": [(X_test_sel, y_test)],
                "verbose": True
            }
        else:
            fit_params = None

        # Anomaly detection (IsolationForest only)
        from sklearn.metrics import classification_report
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=n_pos/(n_pos+n_neg), random_state=42)
        iso.fit(X_train_sel)
        anomaly_pred = iso.predict(X_test_sel)
        anomaly_pred_bin = np.where(anomaly_pred == -1, 1, 0)
        print("\nIsolationForest anomaly detection report:")
        print(classification_report(y_test, anomaly_pred_bin, zero_division=1))
        best_model = iso
        best_model_name = "IsolationForest"

        # Anomaly detection (IsolationForest)
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=n_pos/(n_pos+n_neg), random_state=42)
        iso.fit(X_train_sel)
        anomaly_pred = iso.predict(X_test_sel)
        # IsolationForest: -1 = anomaly, 1 = normal
        # Map to binary: anomaly -> 1 (rockfall), normal -> 0
        anomaly_pred_bin = np.where(anomaly_pred == -1, 1, 0)
        from sklearn.metrics import classification_report
        print("\nIsolationForest anomaly detection report:")
        print(classification_report(y_test, anomaly_pred_bin, zero_division=1))

        # Model evaluation and logging
        y_train_pred = best_model.predict(X_train_sel)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test_sel)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        rockfall_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=rockfall_model)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
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