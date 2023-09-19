from fastapi import FastAPI
from irisModel import IrisMatchineLearning, IrisSpecies

app = FastAPI()
model = IrisMatchineLearning()

@app.get("/")
async def root():
    return {"message": "Hello This is iris classfier"}

@app.post("/predict")
async def predict_species(iris: IrisSpecies):
    pred, prob = model.predict_species(iris.sepal_length, iris.sepal_width,
                                       iris.petal_length, iris.petal_width)
    return {"prediction": pred,
            "probability": prob.tolist()}
