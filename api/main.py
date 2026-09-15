from fastapi import FastAPI
from src.predict import predict
from pydantic import BaseModel, Field

app = FastAPI()


@app.get("/health")
def health():
    return {"res": "App is working...."}


class InputData(BaseModel):
    text: str = "hp laptop with 8gm ram and 500gb hdd"


@app.post("/predict")
def predict_route(input_data: InputData):
    return predict(input_data.text)
