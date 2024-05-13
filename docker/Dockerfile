FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

WORKDIR /root
RUN apt-get update -y && apt-get install -y python3 python3-pip libcudnn8 libcudnn8-dev
COPY infer.py jfk.flac ./
RUN pip3 install faster-whisper

ENTRYPOINT ["python3", "infer.py"]
