from __future__ import annotations

import uvicorn as uvicorn
from fastapi import FastAPI

from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from neural_net.predict_proba import PredictProba


class Proba(BaseModel):
    noise: float
    factor: float
    random_state: int


app = FastAPI()


@app.post("/calculate")
def calculate(proba: Proba):
    obj = PredictProba(noise=proba.noise, factor=proba.factor, random_state=proba.random_state)
    response = obj.calculate()
    del obj
    return JSONResponse(content=jsonable_encoder(response))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
