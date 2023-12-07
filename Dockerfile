FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt ./
COPY yieldestimation.py ./

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 

RUN pip3 install -r requirements.txt

COPY . .

CMD python3 yieldestimation.py