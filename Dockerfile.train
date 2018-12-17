FROM tensorflow/tensorflow:1.12.0-gpu-py3

LABEL maintainer="edhenry@protonmail.com"

COPY . /opt/chexnet

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6 && \
    apt-get install -y libsm6 libxext6 libxrender1 libfontconfig1 libxrender-dev python3.6-dev

RUN apt-get install python3-pip -y

WORKDIR /opt/chexnet
RUN python3.6 -m pip install -r requirements.txt
RUN python3.6 -m pip install opencv-contrib-python
RUN mkdir /var/log/tensorboard

EXPOSE 3000/tcp
EXPOSE 6006/tcp

#ENTRYPOINT [ "python3.6", "train.py" ]
CMD ["nohup python3.6 /opt/chexnet/train.py & \&& tensorboard --logdir /var/log/tensorboard"]
