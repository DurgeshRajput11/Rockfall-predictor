import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.signal import butter, lfilter
import joblib

# Assuming these are defined in your project's constants and utils
from rockfallsecurity.constant.training_pipeline import TARGET_COLUMN
from rockfallsecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from rockfallsecurity.entity.config_entity import DataTransformationConfig
from rockfallsecurity.exception.exception import RockfallSafetyException 
from rockfallsecurity.logging.logger import logging
from rockfallsecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise RockfallSafetyException(e, sys)
            
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise RockfallSafetyException(e, sys)
            
    @staticmethod
    def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies specific noise filters to the sensor data columns.
        """
        logging.info("Applying noise filters to sensor data...")
        df_filtered = df.copy()

        # 1. Moving Average Filter for Pore Water Pressure
        df_filtered['pore_pressure_kpa'] = df_filtered['pore_pressure_kpa'].rolling(window=5, center=True, min_periods=1).mean()
        
        # 2. Bandpass Filter for Ground Acceleration
        fs = 100.0  # Sample frequency (Hz)
        lowcut = 1.0
        highcut = 20.0
        order = 5
        
        try:
            nyq = 0.5 * fs
            low, high = lowcut / nyq, highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            df_filtered['ground_acceleration_g'] = lfilter(b, a, df_filtered['ground_acceleration_g'])
        except Exception as filter_error:
            logging.warning(f"Could not apply bandpass filter. Error: {filter_error}")

        return df_filtered

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a data transformation pipeline object.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numerical_cols = [
                '3d_surface_coord_x', '3d_surface_coord_y', '3d_surface_coord_z',
                'pore_pressure_kpa', 'subsurface_displacement_mm', 'precipitation_rate_mm_hr',
                'ground_acceleration_g', 'realtime_water_table_m', 'cumulative_precip_72hr',
                'cumulative_precip_168hr', 'ucs_mpa', 'pga_g', 'initial_water_table_m'
            ]
            categorical_cols = ['location_id', 'seismic_zone']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipelines", cat_pipeline, categorical_cols),
                ]
            )
            return preprocessor
        except Exception as e:
            raise RockfallSafetyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            # Step 1: Read Data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Step 2: Drop Timestamp
            train_df = train_df.drop(columns=['timestamp'])
            test_df = test_df.drop(columns=['timestamp'])

            # Step 3: Apply Noise Filters
            train_df_filtered = self._apply_filters(train_df)
            test_df_filtered = self._apply_filters(test_df)

            # Step 4: Get Preprocessor
            preprocessor = self.get_data_transformer_object()

            # Step 5: Separate Features and Target
            input_feature_train_df = train_df_filtered.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df_filtered[TARGET_COLUMN]
            input_feature_test_df = test_df_filtered.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df_filtered[TARGET_COLUMN]

            # Step 6: Apply Preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Step 7: Combine and Save
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            
            # As in your original code, save an extra copy for easy access
            final_model_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)
            os.makedirs(final_model_dir, exist_ok=True)
            save_object(os.path.join(final_model_dir, "preprocessor.pkl"), preprocessor)

            # Step 8: Prepare and Return Artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise RockfallSafetyException(e, sys)
