FROM ubuntu:16.04
MAINTAINER Ed Henry "edward.m.henry@gmail.com"

RUN apt-get update -y

RUN apt-get install -y python-pip python-dev build-essential python-qt4
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade

COPY . /app

WORKDIR /app

RUN python3.6 -m pip install -r requirements.txt

ENTRYPOINT ["python3.6"]

CMD ["app.py"]