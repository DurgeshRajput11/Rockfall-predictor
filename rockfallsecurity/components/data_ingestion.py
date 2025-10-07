from rockfallsecurity.exception.exception import RockfallSafetyException
from rockfallsecurity.logging.logger import logging


## configuration of the Data Ingestion Config

from rockfallsecurity.entity.config_entity import DataIngestionConfig
from rockfallsecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise RockfallSafetyException(e,sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb
        """
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            import certifi
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
            collection=self.mongo_client[database_name][collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise RockfallSafetyException(e, sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise RockfallSafetyException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            if dataframe.empty:
                raise RockfallSafetyException("Loaded DataFrame is empty before train/test split!", sys)
            # Prefer time-based split if timestamp exists and is parseable
            test_ratio = self.data_ingestion_config.train_test_split_ratio
            train_set = test_set = None
            used_time_split = False
            # Minimum positives we want to see in test for stable eval
            MIN_TEST_POS = 3
            if 'timestamp' in dataframe.columns:
                df_ts = dataframe.copy()
                df_ts['__ts'] = pd.to_datetime(df_ts['timestamp'], errors='coerce')
                if df_ts['__ts'].notna().any():
                    df_ts = df_ts.sort_values('__ts')
                    # Initial split index based on configured ratio
                    split_idx = max(int(len(df_ts) * (1 - test_ratio)), 1)
                    # Try to ensure a minimum number of positives in test by adaptively moving the boundary earlier
                    def make_splits(idx: int):
                        tr = df_ts.iloc[:idx].drop(columns=['__ts'])
                        te = df_ts.iloc[idx:].drop(columns=['__ts'])
                        return tr, te
                    train_set, test_set = make_splits(split_idx)
                    # If classification target exists, ensure both splits contain positives when possible
                    if 'rockfall_event' in dataframe.columns:
                        def count_pos(df):
                            try:
                                return int((df['rockfall_event'] == 1).sum())
                            except Exception:
                                return 0
                        pos_train = count_pos(train_set)
                        pos_test = count_pos(test_set)
                        # Expand test set (move boundary earlier) until we meet MIN_TEST_POS or hit 50% test size
                        if pos_test < MIN_TEST_POS:
                            # Move split earlier in 5% steps up to 50% test size
                            max_test_frac = 0.5
                            step = max(int(0.05 * len(df_ts)), 1)
                            min_idx = max(int(len(df_ts) * (1 - max_test_frac)), 1)
                            idx = split_idx
                            while idx > min_idx and pos_test < MIN_TEST_POS:
                                idx = max(idx - step, min_idx)
                                tr, te = make_splits(idx)
                                pos_test = count_pos(te)
                                pos_train = count_pos(tr)
                                if pos_test >= MIN_TEST_POS and pos_train > 0:
                                    train_set, test_set = tr, te
                                    break
                        has_pos_train = (pos_train > 0)
                        has_pos_test = (pos_test > 0)
                        if not (has_pos_train and has_pos_test):
                            train_set = None
                            test_set = None
                        else:
                            used_time_split = True
                    else:
                        used_time_split = True
            if train_set is None or test_set is None:
                # Fallback to stratified random split
                stratify_col = None
                if 'rockfall_event' in dataframe.columns:
                    vc = dataframe['rockfall_event'].value_counts(dropna=False)
                    if vc.get(0, 0) > 0 and vc.get(1, 0) > 0:
                        stratify_col = dataframe['rockfall_event']
                train_set, test_set = train_test_split(
                    dataframe,
                    test_size=test_ratio,
                    random_state=42,
                    stratify=stratify_col,
                )
                logging.info("Used stratified random split for train/test.")
            else:
                logging.info("Used time-based split for train/test.")
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise RockfallSafetyException(e, sys)
        
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact
        except Exception as e:
            raise RockfallSafetyException(e, sys)