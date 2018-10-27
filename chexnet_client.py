import numpy as np
import PIL
import cv2
from kafka import KafkaProducer, KafkaClient


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

def do_inference(ts_sever: str, ts_port: int, model_input):
    """
    API call to perform inference over a given input
    
    Arguments:
        ts_sever {str} -- TensorFlow Serving IP
        ts_port {int} -- TensorFlow Serving Port 
        model_input {[type]} -- Input tensor 
    """
    


