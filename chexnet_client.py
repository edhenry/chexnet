import cv2
import grpc
from configparser import ConfigParser
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
import io
import logging
import numpy as np
from PIL import Image
import sys
import tensorflow as tf
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import threading

# TODO explore extending model definition in SavedModel 
# to account for returning a Class Activation Map (CAM)
# for overlay onto xray image that has been uploaded

config_file = "./sample_config.ini"
cp = ConfigParser()
cp.read(config_file)

bootstrap_server = cp["KAFKA"].get("bootstrap_server")
group_id = cp["KAFKA"].get("group_id")
topic = cp["KAFKA"].get("kafka_topic").split(',')
offset = cp["KAFKA"].get("offset_reset")

def logger():
    """Logger instance

        Logs will be emitted when poll() is called when used with Consumer
    
    Returns:
        [logging.Logger] -- Logging object
    """

    logger = logging.getLogger('consumer')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
    logger.addHandler(handler)

    return logger

def connect_to_kafka() -> Consumer:
    """Connect to Kafka broker
    
    Returns:
        Consumer -- return Consumer object
    """

    logs = logger()

    c = Consumer({
        'bootstrap.servers': bootstrap_server,
        'group.id': group_id,
        'auto.offset.reset': offset
    }, logger=logs)

    return c

def do_inference(ts_server: str, ts_port: int, model_input):
    """
    API call to perform inference over a given input
    
    Arguments:
        ts_sever {str} -- TensorFlow Serving IP
        ts_port {int} -- TensorFlow Serving Port 
        model_input {[type]} -- Input tensor 
    """    
    channel = grpc.insecure_channel(ts_server + ":" + str(ts_port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'DenseNet121'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(model_input, dtype=types_pb2.DT_FLOAT, shape=[1, 224, 224, 3])
    )

    result_future = stub.Predict(request, 5.0)

    prediction = tensor_util.MakeNdarray(result_future.outputs['prediction'])
    class_weights = tensor_util.MakeNdarray(result_future.outputs['class_weights'])
    final_conv_layer = tensor_util.MakeNdarray(result_future.outputs['final_conv_layer'])

def collect_image(topic: str, kafka_session: Consumer):
    """Collect an image from the respective image topic
    
    Arguments:
        broker {str} -- Kafka client
        topic {str} -- topic (ex. images)
    """
    
    def print_assignment(consumer, partitions):
        print('Assignment:', partitions)

    kafka_session.subscribe(topic, on_assign=print_assignment)
    
    while True:
        msg = kafka_session.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                sys.stderr.write('%% %s [%d] reached end of offset %d\n' %
                                 (msg.topic(), msg.partition(), msg.offset()))
            else:
                raise KafkaException(msg.error())
        else:
            # Well formed messaged
            sys.stderr.write('%% %s [%d] at offset %d with key %s: \n' %
                             (msg.topic(), msg.partition(), msg.offset(),
                              str(msg.key())))
            
            # image transform
            image_bytes = bytearray(msg.value())
            image = Image.open(io.BytesIO(image_bytes))

            # convert image to array
            image_array = np.asarray(image.convert("RGB"))
            image_array = image_array / 255.
            image_array = np.resize(image_array, (1, 224, 224, 3))

            do_inference(ts_server="172.23.0.9", ts_port=8500, model_input=image_array)

def main():
    logger()
    kafka = connect_to_kafka()
    collect_image(topic, kafka)

if __name__ == '__main__':
    main()