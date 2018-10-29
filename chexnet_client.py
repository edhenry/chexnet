import cv2
import grpc
from kafka import KafkaProducer, KafkaClient
import numpy as np
import PIL
import sys
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import threading

# TODO explore extending model definition in SavedModel 
# to account for returning a Class Activation Map (CAM)
# for overlay onto xray image that has been uploaded

def connect_to_kafka(broker: str, port: int):
    """Connect to Kafka broker
    
    Arguments:
        broker {str} -- broker IP address
        port {int} -- broker port
    """

def collect_image(broker: KafkaClient, topic: str):
    """Collect an image from the respective image topic
    
    Arguments:
        broker {str} -- Kafka client
        topic {str} -- topic (ex. images)
    """

def do_inference(ts_server: str, ts_port: int, model_input, work_dir: str):
    """
    API call to perform inference over a given input
    
    Arguments:
        ts_sever {str} -- TensorFlow Serving IP
        ts_port {int} -- TensorFlow Serving Port 
        model_input {[type]} -- Input tensor 
    """

    
    channel = grpc.insecure_channel(ts_server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'DenseNet121'
    request.model_spec.signiture_name = 'images'
    


