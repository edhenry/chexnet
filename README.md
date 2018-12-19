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

### [Jupyter](https://jupyter.org/)

Project Jupyter exists to develop open-source software, open-standards, and services for interactive computing across dozens of programming languages. This will be used within the project as a UI for uploading images that we would like to run through our model.

### [Pachyderm](http://www.pachyderm.io/open_source.html)(WIP)

Pachyderm is used for data versioning and pipelining, like the link above states. This project leverages Pachyderm to create pipelines that are used not only in the preprocessing required for input images, but also for AB testing.

### [Ansible](https://www.ansible.com/overview/how-ansible-works)

Ansible is an automation framework that can be used to define and provision software environments. We use this to provision the tooling required for the end to end pipeline.

There will be another README at the root of the playbooks directory that outlines what each play accomplishes should anyone want to modify or extend the framework.

### [Docker](https://www.docker.com/why-docker)

Docker is an open source container platform that is great for dependency management, especially in projects with a diverse set of tools and libraries required for production. Docker is leveraged heavily in this example
environment for dependency management and shipping models to `production`.

### [Docker Compose](https://docs.docker.com/compose/overview/)

Docker compose is used to define each of the respective environments for all of the supporting services surrounding the entire `pipeline`. A `docker-compose.yml` can be found under each of the `roles` within the `playbook` directory of this repository.

### [TensorFlow](https://github.com/tensorflow/tensorflow)

This is the popular machine learning library released by Google. This is the library used for defining and training the CheXNet model. 

### [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)

This tool is used for tracking the training of the machine learning model.

### [Kafka](https://www.confluent.io/what-is-apache-kafka/)

Kafka is used as a message bus between the various services that would like to consume the images as they're feed into the system from the UI.

### [TensorFlow Serving](https://www.tensorflow.org/serving/)

TensorFlow Serving is a model server that will be used to serve trained models. API calls will be made against the TensorFlow Server for performing inference over a trained model.