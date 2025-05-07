import sys

import pandas as pd
from src.logger import logging
from src.exception import CustomException


from src.utlis import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path='artifacts/preprocessor.pkl'
            model_path='artifacts/model.pkl'

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,
        age: float,
        sex: int,
        cp: int,
        trestbps: float,
        chol: float,
        fbs: int,
        restecg: int,
        thalach: float,
        exang: int,
        oldpeak: float,
        slope: int,
        ca: int,
        thal: int,
    ):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "cp": [self.cp],
                "trestbps": [self.trestbps],
                "chol": [self.chol],
                "fbs": [self.fbs],
                "restecg": [self.restecg],
                "thalach": [self.thalach],
                "exang": [self.exang],
                "oldpeak": [self.oldpeak],
                "slope": [self.slope],
                "ca": [self.ca],
                "thal": [self.thal]
            }
            import pandas as pd
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Exception(f"Error creating DataFrame from user input: {e}")
