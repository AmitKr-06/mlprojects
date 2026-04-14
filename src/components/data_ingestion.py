import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import DataLoader
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    
    # Add configuration for data loading
    use_config: bool = True  # Set to False to use direct CSV
    csv_path: str = 'notebook/data/titanic_train.csv'  # Default CSV path
    config_path: str = 'config/data_config.yaml'  # Config file path

class DataIngestion:
    def __init__(self, config_path=None):
        self.ingestion_config = DataIngestionConfig()
        self.config_path = config_path or self.ingestion_config.config_path
        
        # Initialize data loader
        if self.ingestion_config.use_config:
            self.data_loader = DataLoader(self.config_path)
        else:
            self.data_loader = None
    
    def initiate_data_ingestion(self, df=None):
        """
        Initiate data ingestion
        
        Args:
            df: Optional DataFrame. If provided, use it directly
        """
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Load data from configured source if not provided
            if df is None:
                if self.data_loader:
                    # Use flexible data loader
                    df = self.data_loader.load_data()
                else:
                    # Fallback to direct CSV
                    df = pd.read_csv(self.ingestion_config.csv_path)
            
            logging.info(f'Read dataset. Shape: {df.shape}')
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")
            
            # Train-test split
            test_size = self.data_loader.config['split_config']['test_size'] if self.data_loader else 0.2
            random_state = self.data_loader.config['split_config']['random_state'] if self.data_loader else 42
            shuffle = self.data_loader.config['split_config']['shuffle'] if self.data_loader else True
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                shuffle=shuffle
            )
            
            # Save splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test with different data sources
    
    # Option 1: Use default CSV
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # Option 2: Use config file
    # obj = DataIngestion(config_path='config/data_config.yaml')
    # train_data, test_data = obj.initiate_data_ingestion()
    
    # Option 3: Pass DataFrame directly
    # df = pd.read_csv('some_data.csv')
    # obj = DataIngestion()
    # train_data, test_data = obj.initiate_data_ingestion(df)
    
    # Continue with data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))