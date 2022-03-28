import io
import uvicorn
import numpy as np
import nest_asyncio
import aiofiles
from enum import Enum
from configs.config import AssetsConfig
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import os
import shutil
import time

from process import *

app = FastAPI(title='API Detect Covid')

call2Engine = Call2Engine()
call2Engine.getToken()

dir_path = os.path.dirname(os.path.realpath(__file__))
tmpPath = os.path.join(dir_path, 'tmp')
if os.path.exists(tmpPath):
    shutil.rmtree(tmpPath)
if not os.path.exists(tmpPath):
    os.mkdir(tmpPath)
    

class Metadata(BaseModel):
    uuid: str = None
    subject_gender: Optional[str] = None
    subject_age: Optional[int] = None
    subject_cough_type: Optional[str] = None
    subject_health_status: Optional[str] = None
    note: Optional[str] = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Author: BunChaBert. Now head over to " \
           "/docs. "


@app.get("/api/v1/news")
def getDataVn():
    return crawlDataCovidVn()


@app.post("/api/v1/uploadImgFirst")
async def uploadImgFirst(uuid: str, fileUpload: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    filename = fileUpload.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    if not isinstance(uuid, str):
        raise HTTPException(status_code=415, detail="id user must be string")

    file_location = f"tmp/{fileUpload.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(fileUpload.file.read())
    print(f"info: file {fileUpload.filename} saved at {file_location}")

    res = img2Emb(uuid, file_location)
    return {
        "result": res,
    }


@app.post("/api/v1/uploadAndPredictVideo")
async def uploadAndPredictVideo(uuid: str, engine: int, fileUpload: UploadFile = File(...)):
    file_location = f"tmp/{fileUpload.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(fileUpload.file.read())
    print(f"info: file {fileUpload.filename} saved at {file_location}")

    # check video that hay khong
    checkIdentify = checkIdeViaVideo(file_location, uuid)

    if checkIdentify == True:
        pathWav, fileWavName = video2Audio(file_location, uuid)
        pred = ""
        if engine == 0:  # predict qua my model
            pred = predict(open(pathWav, 'rb'))
            recommend = responeWithRecommend(pred)
        elif engine == 1: # predict qua engine hackathon
            pred = call2Engine.callApi(fileWavName, open(pathWav, 'rb'))
            pred = handleResponeEngine(pred)
            recommend = responeWithRecommend(pred)
        return {
            "result": pred,
            "recommend": recommend
        }
    else:
        return checkIdentify


@app.post("/api/v1/predictMyModelDetectCovid")
async def predictMyModel(meta: Metadata = Form(...),
                         fileUpload: UploadFile = File(...)):
    
    timestamp = time.time()

    if not meta.uuid:
        raise HTTPException(status_code=404, detail="UUID not found")

    ext = fileUpload.filename.split(".")[-1]
    audio_path = str(
        AssetsConfig.AUDIO_PATH / "{}-{}.{}".format(meta.uuid, timestamp, ext)
    )

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
        pred = predict(fileUpload.file)
    
    async with aiofiles.open(audio_path, "wb") as f:
        content = await fileUpload.read()  # async read
        await f.write(content)  # async write
    
    metadata_json = {
        "uuid": meta.uuid,
        "subject_gender": meta.subject_gender,
        "subject_age": meta.subject_age,
        "subject_cough_type": meta.subject_cough_type,
        "subject_health_status": meta.subject_health_status,
        "note": meta.note,
        "file_path": audio_path,
    }

    metadata_json["result"] = pred
    metadata_json["recommend"] = responeWithRecommend(pred)

    # save metadata
    metadata_path = str(
        AssetsConfig.META_PATH / "{}-{}.json".format(meta.uuid, timestamp)
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata_json, f)
        
    return {
        "result": pred,
        "recommend": responeWithRecommend(pred)
    }


@app.post("/api/v1/predictEngineBGK")
async def predictEngine(meta: Metadata = Form(...),
                        fileUpload: UploadFile = File(...)):
    
    timestamp = time.time()

    if not meta.uuid:
        raise HTTPException(status_code=404, detail="UUID not found")

    ext = fileUpload.filename.split(".")[-1]
    audio_path = str(
        AssetsConfig.AUDIO_PATH / "{}-{}.{}".format(meta.uuid, timestamp, ext)
    )
    
    filename = fileUpload.filename
    fileExtension = filename.split(".")[-1] in ("wav", "m4a")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    if filename.split(".")[-1] in ("m4a"):
        file_location = f"tmp/{fileUpload.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(fileUpload.file.read())
        # print(f"info: file {fileUpload.filename} saved at {file_location}")
        pathWav, filename = m4aToWav(file_location, fileUpload.filename)
        pred = call2Engine.callApi(filename, open(pathWav, 'rb'))
    else:
        pred = call2Engine.callApi(fileUpload.filename, fileUpload.file)

    async with aiofiles.open(audio_path, "wb") as f:
        content = await fileUpload.read()  # async read
        await f.write(content)  # async write
    
    metadata_json = {
        "uuid": meta.uuid,
        "subject_gender": meta.subject_gender,
        "subject_age": meta.subject_age,
        "subject_cough_type": meta.subject_cough_type,
        "subject_health_status": meta.subject_health_status,
        "note": meta.note,
        "file_path": audio_path,
    }
    metadata_json["result"] = handleResponeEngine(pred)
    metadata_json["recommend"] = responeWithRecommend(handleResponeEngine(pred))

    # save metadata
    metadata_path = str(
        AssetsConfig.META_PATH / "{}-{}.json".format(meta.uuid, timestamp)
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata_json, f)

    return {
        "result": handleResponeEngine(pred),
        "recommend": responeWithRecommend(handleResponeEngine(pred))
    }


# Allows the server to be run in this interactive environment
nest_asyncio.apply()
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"
uvicorn.run("serve:app", host="0.0.0.0")

