from __future__ import annotations

import uvicorn as uvicorn
from fastapi import FastAPI

from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from csv_manager.csv_manager import CsvManager
from neural_net.predict_proba import PredictProba


class Proba(BaseModel):
    noise: float
    factor: float
    random_state: int


class File(BaseModel):
    file_name: str


app = FastAPI()


@app.post("/calculate")
def calculate(proba: Proba):
    obj = PredictProba(noise=proba.noise, factor=proba.factor, random_state=proba.random_state)
    response = obj.calculate()
    del obj
    return JSONResponse(content=jsonable_encoder(response))


@app.post("/calculate_via_file")
def calculate_via_file(file: File):
    result = []
    csv = CsvManager(file_name=f"csv/{file.file_name}")
    data = csv.get_data()
    for i in data:
        obj = PredictProba(noise=float(i["noise"]), factor=float(i["factor"]), random_state=int(i['random_state']))
        result.append(obj.calculate())
    csv.create_append_data_to_csv(result, headers=["array_form", "decision_func_form", "decision_func", "decision_func_threshold", "predict"])
    return JSONResponse({"status": "success"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
