FROM tensorflow/tensorflow:1.11.0-gpu-py3

LABEL maintainer="edhenry@protonmail.com"

COPY . /opt/chexnet

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6 && \
    apt-get install -y libsm6 libxext6 libxrender1 libfontconfig1 libxrender-dev python3.6-dev

RUN apt-get install python3-pip -y
RUN apt-get install python3.6-tk -y

WORKDIR /opt/chexnet
RUN python3.6 -m pip install -r requirements.txt

CMD ["python3.6", "chexnet_client.py"]