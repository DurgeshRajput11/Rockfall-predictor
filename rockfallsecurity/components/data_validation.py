
import pandas as pd
from scipy.stats import ks_2samp
import os
import sys

# Assuming these are defined in your project's structure
from rockfallsecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from rockfallsecurity.entity.config_entity import DataValidationConfig
from rockfallsecurity.exception.exception import RockfallSafetyException 
from rockfallsecurity.logging.logger import logging 
from rockfallsecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from rockfallsecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Initializes the DataValidation component.

        Args:
            data_ingestion_artifact: Artifact from the Data Ingestion stage.
            data_validation_config: Configuration for the Data Validation stage.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            # Read the schema which defines the expected data structure
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            if "columns" not in self._schema_config:
                raise Exception("'columns' key missing in schema config.")
            if "numerical_columns" not in self._schema_config:
                raise Exception("'numerical_columns' key missing in schema config.")
        except Exception as e:
            raise RockfallSafetyException(f"Error initializing DataValidation: {e}", sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads a CSV file into a Pandas DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise RockfallSafetyException(f"Error reading data from {file_path}: {e}", sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates that the number of columns in the DataFrame matches the schema.
        """
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Schema requires {number_of_columns} columns.")
            logging.info(f"DataFrame has {len(dataframe.columns)} columns.")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise RockfallSafetyException(f"Error validating number of columns: {e}", sys)
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Detects data drift between a base (training) and current (testing) DataFrame.
        Uses the two-sample Kolmogorov-Smirnov (KS) test.
        """
        logging.info("Starting dataset drift detection...")
        try:
            validation_status = True
            drift_report = {}
            numerical_columns = self._schema_config["numerical_columns"]
            for column in numerical_columns:
                if column not in base_df.columns or column not in current_df.columns:
                    logging.warning(f"Column '{column}' not found in both dataframes for drift detection.")
                    continue
                d1 = base_df[column]
                d2 = current_df[column]
                ks_statistic, p_value = ks_2samp(d1, d2)
                is_found = p_value < threshold
                if is_found:
                    validation_status = False
                drift_report[column] = {
                    "p_value": float(p_value),
                    "drift_status": is_found
                }
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=drift_report)
            logging.info(f"Drift detection complete. Report saved to: {drift_report_file_path}")
            return validation_status
        except Exception as e:
            raise RockfallSafetyException(f"Error in dataset drift detection: {e}", sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        The main method to orchestrate the entire data validation process.
        """
        logging.info("--- Starting Data Validation ---")
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            # Read data
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)
            # 1. Schema Validation (Number of Columns)
            logging.info("Validating number of columns...")
            is_train_valid = self.validate_number_of_columns(dataframe=train_dataframe)
            if not is_train_valid:
                raise RockfallSafetyException("Training DataFrame does not match the schema's column count.", sys)
            is_test_valid = self.validate_number_of_columns(dataframe=test_dataframe)
            if not is_test_valid:
                raise RockfallSafetyException("Testing DataFrame does not match the schema's column count.", sys)
            # 2. Data Drift Detection
            validation_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            if validation_status:
                logging.info("No significant data drift detected between train and test sets.")
            else:
                logging.warning("Data drift detected! Check the drift report for details.")
            # If all validations pass, save the validated data to new artifact paths
            valid_train_path = self.data_validation_config.valid_train_file_path
            valid_test_path = self.data_validation_config.valid_test_file_path
            os.makedirs(os.path.dirname(valid_train_path), exist_ok=True)
            train_dataframe.to_csv(valid_train_path, index=False, header=True)
            test_dataframe.to_csv(valid_test_path, index=False, header=True)
            # Prepare and return the output artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                invalid_train_file_path=None,  # Assuming success
                invalid_test_file_path=None,   # Assuming success
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            logging.info("--- Data Validation Completed ---")
            return data_validation_artifact
        except Exception as e:
            raise RockfallSafetyException(f"Error in initiate_data_validation: {e}", sys)
