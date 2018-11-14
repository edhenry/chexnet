import cv2
import grpc
from configparser import ConfigParser
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
import generator
import io
import json
import keras.backend as K
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import scipy.misc
from skimage.transform import resize
from io import StringIO
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
bootstrap_port = cp["KAFKA"].get("bootstrap_port")
group_id = cp["KAFKA"].get("group_id")
inference_kafka_topic = cp["KAFKA"].get("inference_kafka_topic").split(',')
results_kafka_topic = cp["KAFKA"].get("results_kafka_topic")
offset = cp["KAFKA"].get("offset_reset")
class_names = cp["DEFAULT"].get("class_names").split(",")

def logger():
    """Logger instance

        Logs will be emitted when poll() is called when used with Consumer and/or Producer
    
    Returns:
        [logging.Logger] -- Logging object
    """

    logger = logging.getLogger('chexnet_client')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    return logger

logs = logger()

def kafka_consumer() -> Consumer:
    """Connect and consume data from Kafka Broker
    
    Returns:
        Consumer -- return Consumer object
    """

    c = Consumer({
        'bootstrap.servers': bootstrap_server,
        'group.id': group_id,
        'auto.offset.reset': offset
    }, logger=logs)

    return c

def kafka_producer() -> Producer:
    """Connect and publish data to Kafka broker
    
    Returns:
        Producer -- [description]
    """

    p = Producer({
        'bootstrap.servers': bootstrap_server,
        'message.max.bytes': 10000000
    }, logger=logs)

    return p

def kafka_delivery_report(err, msg):
    """Called once for each messaged produced to indicate delivery result

    Triggered by poll() or flush()
    """
    if err is not None:
        logs.info('Message delivery failed! : {}'.format(err))
    else:
        logs.info('Message delivered to {} [{}] at offset [{}]'.format(msg.topic(), msg.partition(), msg.offset()))

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

    logs.info("Successfully received response from TensorFlow Server!")

    return prediction, class_weights, final_conv_layer

def image_transform(msg_payload) -> Image:
    """Transform message from Kafka message payload
    
    Arguments:
        msg_payload {Consumer.poll} -- message payload

    Returns:
        PIL.Image -- Image object
    """

    image_bytes = bytearray(msg_payload.value())
    image = Image.open(io.BytesIO(image_bytes))

    orig_image_array = np.asarray(image.convert("RGB"))
    image_array = orig_image_array / 255.
    image_array = resize(image_array, (1, 224, 224, 3))
    logs.info("topic : [%s] - offset : [%s] - image successfully transformed!", msg_payload.topic(), msg_payload.offset())

    return image_array, orig_image_array

def marshall_message(img_bytes, aurocs) -> dict:
    """Marshall message to send over message bus

       In the future I would rather use something like Protobufs / Avro instead of 
       raw JSON
    
    Arguments:
        img_bytes {bytearray} -- byte array to convert to string for transmission
        aurocs {numpy array} -- numpy array of prediction results
    
    Returns:
        dict -- [description]
    """

    ser_message = {}

    img_bytes = img_bytes.decode('latin-1')

    ser_message['image'] = img_bytes
    ser_message['aurocs'] = aurocs

    return json.dumps(ser_message)

def create_barchart(prediction_array):
    """Create a barchart for predictions
    
    Arguments:
        prediction_array {numpy array} -- Array of predictions returned from CheXNet Model
    """
    y_pos = class_names

    plt.barh(y_pos, prediction_array, align='center', alpha=0.5)
    plt.yticks(y_pos, class_names)
    plt.xlabel('Probability')
    plt.title("Probability of given pathology")
    plt.savefig("barchart.png")

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
            logs.info("No messages available within topic : %s", topic)
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                logs.info('%% %s [%d] reached end of offset %d' %
                                 (msg.topic(), msg.partition(), msg.offset()))
            else:
                logs.debug("Kafka Exception : %s", msg.error())
                raise KafkaException(msg.error())
        else:
            # Well formed messaged
            logs.info('%% %s [%d] at offset %d with key %s: ' %
                             (msg.topic(), msg.partition(), msg.offset(),
                              str(msg.key())))
            
            # image transform
            image_array, orig_image_array = image_transform(msg)

            prediction, class_weights, final_conv_layer = do_inference(ts_server="172.23.0.9", ts_port=8500, model_input=image_array)

            # create CAM
            get_output = K.function([tf.convert_to_tensor(image_array)], [tf.convert_to_tensor(final_conv_layer), tf.convert_to_tensor(prediction)])
            [conv_outputs, predictions] = get_output([image_array[0]])
            conv_outputs = conv_outputs[0, :, :, :]

            # TODO: Receiving variable results across CAMs generated by this
            # method. Needs further investigation and comparison to original
            # CAM paper found here : http://cnnlocalization.csail.mit.edu/
            cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
            for i, w in enumerate(class_weights[0]):
                cam += w * conv_outputs[:, :, i]
            cam /= np.max(cam)
            cam = cv2.resize(cam, orig_image_array.shape[:2])
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap[np.where(cam < 0.2)] = 0
            img = heatmap * 0.5 + orig_image_array

            logs.info("Class Activation Map (CAM) Created!")

            # this is complete fucking hackery and will need to be replaced
            # I don't know why a numpy array (see `img` array above) would be 25MB when all constituent
            # arrays are ~ 7MB total. Let alone when saving an image to disk
            # the image is only 1MB total. What the actual fuck.
            cv2.imwrite("inflight_img.png", img)

            new_img = Image.open("inflight_img.png", mode='r')
            img_bytes = io.BytesIO()
            new_img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            message = marshall_message(img_bytes, prediction.tolist())
            os.remove("inflight_img.png")

            create_barchart((prediction[0] * 100))

            p = kafka_producer()
            p.poll(0)
            p.produce(results_kafka_topic, value=message, callback=kafka_delivery_report)
            p.flush()
                        
def main():
    # TODO: Restructure execution logic and break apart more
    # complex functions such as collect_image(), etc.
    # KISS and DRY should be applied...

    kafka = kafka_consumer()
    collect_image(inference_kafka_topic, kafka)

if __name__ == '__main__':
    main()