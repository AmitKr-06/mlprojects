import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a preprocessor object for Titanic dataset
        """
        try:
            # Define columns for Titanic dataset
            numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch']
            categorical_columns = ['Name', 'Sex', 'Embarked']
            
            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combine pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Target column for Titanic
            target_column_name = "Survived"
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features with target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info(f"Transformed train array shape: {train_arr.shape}")
            logging.info(f"Transformed test array shape: {test_arr.shape}")
            
            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the data transformation
    obj = DataTransformation()
    
    # Test with sample paths (you can modify these)
    train_path = 'artifacts/train.csv'
    test_path = 'artifacts/test.csv'
    
    # Uncomment to run
    # train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(train_path, test_path)
    # print(f"Train array shape: {train_arr.shape}")
    # print(f"Test array shape: {test_arr.shape}")