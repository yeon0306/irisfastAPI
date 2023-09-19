import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel

class IrisSpecies(BaseModel):
    sepal_length:float
    sepal_width: float
    petal_length:float
    petal_width: float


class IrisMatchineLearning:
    def __init__(self):
        self.iris_df = pd.read_csv("data/iris.csv")
        self.rfc_fname = 'model/model_rfc.pkl'
        try:
            self.model_rfc = joblib.load(self.rfc_fname)
        except Exception as _ :
            self.model_rfc = self.rfc_train()
            joblib.dump(self.rfc_model, self.rfc_fname)


    def rfc_train(self):
        X = self.iris_df.drop('species', axis=1)
        y = self.iris_df['species']
        rfc = RandomForestClassifier()
        model = rfc.fit(X, y)
        return model

    def predict_species(self, sl, sw, pl, pw):
        X_new = np.array([[sl, sw, pl, pw]])
        prediction = self.model_rfc.predict(X_new)
        probability = self.model_rfc.predict_proba(X_new).max()
        print(f'{prediction}')
        print(f'{probability}')
        return prediction[0], probability
