FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

WORKDIR /root

RUN apt-get update -y && apt-get install -y python3-pip

COPY requirements.txt infer.py jfk.flac ./

RUN pip3 install -r requirements.txt

CMD ["python3", "infer.py"]
