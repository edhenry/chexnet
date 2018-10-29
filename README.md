# CheXNet Pipeline

This repository is an implementation of the CheXNet solution outlined in [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](http://arxiv.org/abs/1711.05225).

# Overview

The goal of this implementation and the surrounding tooling to enable someone whom is interested, generally, in Machine Learning (ML) and would like to understand
better what an example end to end Machine Learning pipeline might look like. All of the tools and utilities that are used within this implementation
are open source, highlighting the `open source` and `open science` approach that I think is necessary to allow for further adoption of ML solutions.

The pipeline is general enough that one can swap the `input` portion and `model` portion of the pipelines to allow for other applications. This example is
specific to computer vision, however one can feasibly "slot in" another application using the same tooling and rough outline of a pipeline.

## Tools and utilities

The pipeline consists of many different tools and utilities. I will provide an outline of each tool and utility below and what they are used for.

I will work through the list starting with data acquisition on through to training and deployment of a machine learning model. I will also cover the
process of re-training a model and performing AB testing on the two models to measure whether or not a new model has better performance.

### [Pachyderm]

### [Ansible]

### [Docker]

### [Docker Compose]

### [TensorFlow](https://github.com/tensorflow/tensorflow)

This is the popular machine learning library released by Google. This is the library used for defining and training the CheXNet model. 

### [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)

This tool is used for tracking the training of the machine learning model.

### [Kafka]

### [Airflow]

### [TensorFlow Serving]