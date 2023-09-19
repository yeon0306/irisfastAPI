import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

iris_df = pd.read_csv('data/iris.csv')

X = iris_df.drop('species', axis=1)
Y = iris_df['species']

kn = KNeighborsClassifier()
model_kn = kn.fit(X, Y)

rfc = RandomForestClassifier()
model_rfc = rfc.fit(X, Y)

# 모델 저장
joblib.dump(model_kn, 'model/model_kn.pkl')
joblib.dump(model_rfc, 'model/model_rfc.pkl')

model_kn = joblib.load('model/model_kn.pkl')
model_rfc = joblib.load('model/model_rfc.pkl')

X_new = np.array([[4,4,4,4]])
prediction = model_kn.predict(X_new)
probability = model_kn.predict_proba(X_new).max()
print(f'예측은 {prediction}')
print(f'예측확률은 {probability}')

prediction = model_rfc.predict(X_new)
probability = model_rfc.predict_proba(X_new).max()
print(f'rfc 예측은 {prediction}')
print(f'rfc 예측확률은 {probability}')
