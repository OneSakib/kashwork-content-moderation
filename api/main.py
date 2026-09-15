from fastapi import FastAPI
from src.predict import predict
from pydantic import BaseModel, Field

app = FastAPI()


@app.get("/health")
def health():
    return {"res": "App is working...."}


class InputData(BaseModel):
    text: str = "hp laptop with 8gm ram and 500gb hdd"


class PredictResponse(BaseModel):
    abusive: int
    abusive_score: float
    restricted: int
    restricted_score: float


@app.post("/predict", response_model=PredictResponse)
def predict_route(input_data: InputData) -> PredictResponse:
    predicted = predict(input_data.text)
    response = PredictResponse(
        abusive=predicted["abusive"],
        abusive_score=predicted["abusive_score"],
        restricted=predicted["restricted"],
        restricted_score=predicted["restricted_score"],
    )
    return response
