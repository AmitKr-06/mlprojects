import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def run_pipeline(self):
        """
        Execute the complete training pipeline
        """
        try:
            logging.info("="*50)
            logging.info("Starting Training Pipeline")
            logging.info("="*50)
            
            # Step 1: Data Ingestion
            logging.info("\n--- Step 1: Data Ingestion ---")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Train data saved at: {train_path}")
            logging.info(f"Test data saved at: {test_path}")
            
            # Step 2: Data Transformation
            logging.info("\n--- Step 2: Data Transformation ---")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")
            
            # Step 3: Model Training
            logging.info("\n--- Step 3: Model Training ---")
            model_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Best model score: {model_score:.4f}")
            
            logging.info("\n" + "="*50)
            logging.info("Training Pipeline Completed Successfully!")
            logging.info("="*50)
            
            return model_score
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Run the pipeline
    pipeline = TrainPipeline()
    result = pipeline.run_pipeline()
    print(f"\n Pipeline Completed! Best Model Score: {result:.4f}")