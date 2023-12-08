FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY yieldestimation.py .
COPY data.csv .

COPY model ./model
COPY templates ./templates
COPY test_images ./test_images

RUN mkdir -p ./uploads

CMD python3 ./yieldestimation.py