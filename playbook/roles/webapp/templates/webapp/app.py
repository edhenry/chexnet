import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from kafka import KafkaProducer, KafkaClient
import configparser
import cv2
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
topic = str(cp["KAFKA"].get("kafka_topic"))
print(topic)

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

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
        publish_to_kafka(filename, producer, topic)
    else:
        file_url = None
    return render_template('/index.html', form=form, file_url=file_url)

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