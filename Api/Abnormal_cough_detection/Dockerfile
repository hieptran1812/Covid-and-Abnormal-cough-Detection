FROM python:3.8

WORKDIR /app

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y 
RUN apt-get install ffmpeg

RUN pip install --no-cache-dir joblib==1.1.0 \
uvicorn==0.15.0 \
imutils==0.5.2 \
pydub==0.25.1 \
librosa==0.8.1 \
numpy==1.17.3 \
fastapi==0.75.0 \
nest-asyncio==1.5.1 \
python-multipart \
tensorflow \
ffprobe

COPY . /app

# RUN pip install --no-cache-dir  --upgrade -r /app/requirements.txt

EXPOSE 8001
CMD python3 serve.py
