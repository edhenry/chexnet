FROM tensorflow/tensorflow:1.11.0-gpu-py3

COPY . /opt/chexnet

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6

RUN apt-get install python3-pip -y

WORKDIR /opt/chexnet
RUN python3.6 -m pip install -r requirements.txt

EXPOSE 3000/tcp
EXPOSE 6006/tcp

ENTRYPOINT [ "python3.6", "train.py" ]
CMD ["tensorboard --log-dir=/var/log/tensorboard"]
