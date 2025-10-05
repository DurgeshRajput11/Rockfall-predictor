import sys
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Assuming these are defined in your project's structure
from rockfallsecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from rockfallsecurity.entity.config_entity import ModelTrainerConfig
from rockfallsecurity.exception.exception import RockfallSafetyException 
from rockfallsecurity.logging.logger import logging 
from rockfallsecurity.utils.main_utils.utils import save_object

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer component.

        Args:
            data_transformation_artifact: Artifact from the Data Transformation stage.
            model_trainer_config: Configuration for the Model Trainer stage.
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise RockfallSafetyException(e, sys)

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """
        Trains the XGBoost model and returns the trained model object.
        """
        try:
            logging.info("Calculating scale_pos_weight for handling imbalanced data.")
            # This is crucial for imbalanced datasets
            # It tells the model to penalize errors on the minority class (rockfall=1) more heavily
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count
            logging.info(f"Scale Pos Weight: {scale_pos_weight:.2f}")

            # Initialize XGBoost Classifier with hyperparameters
            xgb_clf = XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                random_state=42
            )
            
            logging.info("Starting model training...")
            xgb_clf.fit(x_train, y_train)
            logging.info("Model training completed.")
            
            return xgb_clf
        except Exception as e:
            raise RockfallSafetyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        The main method to orchestrate the model training and evaluation process.
        """
        logging.info("--- Starting Model Trainer ---")
        try:
            # 1. Load the transformed training and testing data
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

            # 2. Split data into features (X) and target (y)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # 3. Train the model
            model = self.train_model(x_train=x_train, y_train=y_train)

            # 4. Evaluate the model on the training set
            y_train_pred = model.predict(x_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            logging.info(f"Model Accuracy on Training Set: {train_accuracy:.4f}")

            # 5. Evaluate the model on the testing set
            y_test_pred = model.predict(x_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            logging.info(f"Model Accuracy on Testing Set: {test_accuracy:.4f}")

            # Check for overfitting
            if train_accuracy > self.model_trainer_config.expected_accuracy + self.model_trainer_config.overfitting_underfitting_threshold:
                 logging.warning("Model may be overfitting. Training accuracy is much higher than expected.")
            
            # Check for underfitting/low performance
            if test_accuracy < self.model_trainer_config.expected_accuracy:
                 raise Exception(f"Model is not good enough. "
                                 f"Expected accuracy: {self.model_trainer_config.expected_accuracy}, "
                                 f"Actual accuracy: {test_accuracy}")

            logging.info("--- Classification Report ---")
            report = classification_report(y_test, y_test_pred)
            logging.info(f"\n{report}")

            # 6. Save the trained model
            save_object(self.model_trainer_config.trained_model_file_path, model)

            # 7. Prepare and return the output artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info("--- Model Trainer Completed ---")
            return model_trainer_artifact

        except Exception as e:
            raise RockfallSafetyException(e, sys)
