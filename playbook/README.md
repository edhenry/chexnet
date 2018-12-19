# What is this playbook?

This playbook is used to build a "production" machine learning training and serving pipelines. All of the roles defined within the `roles` directory are self contained instances of [docker-compose](https://docs.docker.com/compose/overview/) driven services. All of the services are outlined below.

## Quickstart for utilizing this playbook 

### Supported Platforms

When using the term "supported" it is meant to be understood that this playbook and these plays have all been tested on Ubuntu 16.05

* Ubuntu 16.04 LTS (Xenial)

## Services

### [Jupyter](https://jupyter.org/)

Project Jupyter exists to develop open-source software, open-standards, and services for interactive computing across dozens of programming languages. This will be used within the project as a UI for uploading images that we would like to run through our model.

### [Pachyderm](http://www.pachyderm.io/open_source.html)(WIP)

Pachyderm is used for data versioning and pipelining, like the link above states. This project leverages Pachyderm to create pipelines that are used not only in the preprocessing required for input images, but also for AB testing.

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