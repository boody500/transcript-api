FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip git ffmpeg && \
    apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["bash", "startup.sh"]
