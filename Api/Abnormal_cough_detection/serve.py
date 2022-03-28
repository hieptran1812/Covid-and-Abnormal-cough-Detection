import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import librosa
import os
import shutil

from process import *

app = FastAPI(title='API Model Abnormal cough detection')

dir_path = os.path.dirname(os.path.realpath(__file__))
tmpPath = os.path.join(dir_path, 'tmp')
if os.path.exists(tmpPath):
    shutil.rmtree(tmpPath)
if not os.path.exists(tmpPath):
    os.mkdir(tmpPath)

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Author: BunChaBert. Now head over to /docs" 


@app.post("/api/v1/predict")
async def prediction(fileUpload: UploadFile = File(...)): # Upload file tu may tinh
    filename = fileUpload.filename
    fileExtension = filename.split(".")[-1] in ("wav", "m4a")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    if filename.split(".")[-1] in ("m4a"): 
        file_location = f"tmp/{fileUpload.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(fileUpload.file.read())
        print(f"info: file {fileUpload.filename} saved at {file_location}")
        pathWav, _ = m4aToWav(file_location, fileUpload.filename)
        pred = predict(open(pathWav, 'rb'))
    else:
        track = AudioSegment.from_file(fileUpload.file) 
        track_cut = track[500:1500]
        tmpPath = os.path.join(dir_path, 'tmp')
        pathWav = os.path.join(tmpPath, fileUpload.filename)
        track_cut.export(pathWav, format='wav')
        print(f"info: file {fileUpload.filename} saved at {pathWav}")
        pred = predict(open(pathWav, 'rb'))

    return {"result": pred}


if __name__ == '__main__':
  # Allows the server to be run in this interactive environment
  nest_asyncio.apply()

  # Host depends on the setup you selected (docker or virtual env)
  host = "127.0.0.1"

  # Spin up the server!
  uvicorn.run("serve:app", host="0.0.0.0", port=8001)
