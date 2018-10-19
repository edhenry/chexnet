FROM tensorflow/tensorflow:1.11.0-gpu-py3

COPY . src/
RUN pip install --no-cache ptvsd
EXPOSE 3000