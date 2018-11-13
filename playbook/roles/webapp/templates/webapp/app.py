import os
from flask import Flask, render_template, request, Response
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from kafka import KafkaProducer, KafkaClient
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
import configparser
import cv2
import logging
import sys
import time
from PIL import Image
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = "Arthur_Clarke"
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()

config_file = 'webapp_settings.ini'
cp = configparser.ConfigParser()
cp.read(config_file)

broker = cp.get("KAFKA", "broker")
broker_port = cp.get("KAFKA", "kafka_port")
producer = KafkaProducer(bootstrap_servers=(broker.strip('"') + ":" + broker_port))
inference_kafka_topic = str(cp["KAFKA"].get("inference_kafka_topic"))
results_kafka_topic = cp["KAFKA"].get("results_kafka_topic").split(",")
results_consumer_group_id = cp["KAFKA"].get("results_consumer_group_id")
offset_reset = cp["KAFKA"].get("offset_reset")

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

def print_assignment(consumer, partitions):
        print('Assignment:', partitions)

c = Consumer({
    'bootstrap.servers': broker,
    'group.id': results_consumer_group_id,
    'auto.offset.reset': offset_reset
})

c.subscribe(results_kafka_topic, on_assign=print_assignment)

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Please upload images only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Upload file to web server
    """
    
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        publish_to_kafka(filename, producer, inference_kafka_topic)
    else:
        file_url = None
    return render_template('/index.html', form=form, file_url=file_url)

@app.route('/results', methods=['GET'])
def get_results():
    """
    Get results from the model
    """
    collect_results()
    results = "test_img.png"
    return render_template('/results.html', results_url=results)

def collect_results():
    
    while True:
        msg = c.poll(0)
        if msg is None:
            print("No new messages!")
            return
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
            img_bytes = bytearray(msg.value())
            image = Image.open(io.BytesIO(img_bytes))
            image.save("static/test_img.png")
            return
    return 

def publish_to_kafka(image, producer, topic: str):
    """
    
    Publish image to Kafka broker
    
    Arguments:
        image {image file} -- image that has been uploaded to the framework
        producer {KafkaProducer} -- Kafka's native python KafkaProducer
        topic {str} -- Topic to publish the data to
    """

    # convert image to bytes for publishing to Kafka
    img = Image.open(image, mode='r')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    producer.send(topic, img_bytes)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9220)