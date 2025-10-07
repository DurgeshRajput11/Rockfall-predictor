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
        # 1. Moving Average Filter for Pore Water Pressure (causal to avoid leakage)
        df_filtered['pore_pressure_kpa'] = df_filtered['pore_pressure_kpa'].rolling(window=5, center=False, min_periods=1).mean()
        
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

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simple temporal/rolling features per location to improve signal for rare events.
        Assumes rows are roughly time-ordered; uses groupby on location_id and rolling by row-count windows.
        """
        out = df.copy()
        # Parse timestamp if available for stable ordering
        if 'timestamp' in out.columns:
            out['__ts'] = pd.to_datetime(out['timestamp'], errors='coerce')
        else:
            out['__ts'] = pd.NaT

        # Sort within location by timestamp if available
        sort_cols = ['location_id'] + (['__ts'] if out['__ts'].notna().any() else [])
        out = out.sort_values(sort_cols) if sort_cols else out

        def _roll(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            # Basic dynamics
            g['realtime_minus_initial_water'] = g.get('realtime_water_table_m', 0) - g.get('initial_water_table_m', 0)
            g['disp_velocity'] = g['subsurface_displacement_mm'].diff()
            g['disp_acceleration'] = g['disp_velocity'].diff()
            # Rolling windows (row-based) ~ 6 and 24 steps
            for col, base in [
                ('precipitation_rate_mm_hr', 'precipitation_rate'),
                ('pore_pressure_kpa', 'pore_pressure'),
                ('subsurface_displacement_mm', 'subsurface_displacement'),
                ('ground_acceleration_g', 'ground_acceleration'),
            ]:
                if col in g.columns:
                    # Means
                    g[f'{base}_roll6'] = g[col].rolling(window=6, min_periods=1).mean()
                    g[f'{base}_roll24'] = g[col].rolling(window=24, min_periods=1).mean()
                    # Standard deviations
                    g[f'{base}_std6'] = g[col].rolling(window=6, min_periods=2).std()
                    g[f'{base}_std24'] = g[col].rolling(window=24, min_periods=2).std()
                    # Maxima
                    g[f'{base}_max6'] = g[col].rolling(window=6, min_periods=1).max()
                    g[f'{base}_max24'] = g[col].rolling(window=24, min_periods=1).max()
                    # Lags
                    g[f'{base}_lag1'] = g[col].shift(1)
                    g[f'{base}_lag6'] = g[col].shift(6)
                    # Z-score against 24-step window
                    _mu24 = g[f'{base}_roll24']
                    _sd24 = g[f'{base}_std24']
                    g[f'{base}_z24'] = (g[col] - _mu24) / (_sd24.replace(0, np.nan))
            return g

        if 'location_id' in out.columns:
            # Avoid FutureWarning: exclude grouping columns during apply when possible
            try:
                out = out.groupby('location_id', group_keys=False, include_groups=False).apply(_roll)
            except TypeError:
                # Fallback for older pandas without include_groups: drop grouping column before apply and add back
                out = out.groupby('location_id', group_keys=False).apply(
                    lambda g: _roll(g.drop(columns=['location_id'], errors='ignore')).assign(location_id=g['location_id'].iloc[0])
                )
        else:
            out = _roll(out)

        # Keep for imputation, drop helper ts
        out = out.drop(columns=['__ts'], errors='ignore')
        return out

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
            # Engineered features
            numerical_cols += [
                'realtime_minus_initial_water', 'disp_velocity', 'disp_acceleration',
                # Means
                'precipitation_rate_roll6', 'precipitation_rate_roll24',
                'pore_pressure_roll6', 'pore_pressure_roll24',
                'subsurface_displacement_roll6', 'subsurface_displacement_roll24',
                'ground_acceleration_roll6', 'ground_acceleration_roll24',
                # Standard deviations
                'precipitation_rate_std6', 'precipitation_rate_std24',
                'pore_pressure_std6', 'pore_pressure_std24',
                'subsurface_displacement_std6', 'subsurface_displacement_std24',
                'ground_acceleration_std6', 'ground_acceleration_std24',
                # Maxima
                'precipitation_rate_max6', 'precipitation_rate_max24',
                'pore_pressure_max6', 'pore_pressure_max24',
                'subsurface_displacement_max6', 'subsurface_displacement_max24',
                'ground_acceleration_max6', 'ground_acceleration_max24',
                # Lags
                'precipitation_rate_lag1', 'precipitation_rate_lag6',
                'pore_pressure_lag1', 'pore_pressure_lag6',
                'subsurface_displacement_lag1', 'subsurface_displacement_lag6',
                'ground_acceleration_lag1', 'ground_acceleration_lag6',
                # Z-scores (24-step)
                'precipitation_rate_z24', 'pore_pressure_z24',
                'subsurface_displacement_z24', 'ground_acceleration_z24',
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

            # Step 2: Feature Engineering (before dropping timestamp)
            train_df = self._engineer_features(train_df)
            test_df = self._engineer_features(test_df)
            # Drop Timestamp
            train_df = train_df.drop(columns=['timestamp'], errors='ignore')
            test_df = test_df.drop(columns=['timestamp'], errors='ignore')

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

            # Save group labels (location_id) for grouped CV if available
            try:
                transformed_dir = os.path.dirname(self.data_transformation_config.transformed_train_file_path)
                os.makedirs(transformed_dir, exist_ok=True)
                if 'location_id' in input_feature_train_df.columns:
                    train_groups = input_feature_train_df['location_id'].to_numpy()
                    save_numpy_array_data(os.path.join(transformed_dir, 'train_groups.npy'), array=train_groups)
                if 'location_id' in input_feature_test_df.columns:
                    test_groups = input_feature_test_df['location_id'].to_numpy()
                    save_numpy_array_data(os.path.join(transformed_dir, 'test_groups.npy'), array=test_groups)
            except Exception as _err:
                logging.warning(f"Could not save group labels for grouped CV: {_err}")
            
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
