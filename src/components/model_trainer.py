import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_obj,evaluate_models

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")
    
class ModelTraining:
    def __init__(self):
        self.model_training_config=ModelTrainingConfig()
        
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Spliting train ans test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "GradientBoosting":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor()
            }
            
            model_report:dict=evaluate_models(X_train=X_train,
                                             y_train=y_train,
                                             X_test=X_test,
                                             y_test=y_test,
                                             models=models)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best model found on both testing and training dataset")
            
            save_obj(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_sq=r2_score(y_test,predicted)
            print(r2_sq)
        
        except Exception as e:
            raise CustomException(e,sys)