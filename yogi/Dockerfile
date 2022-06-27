FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev libpq-dev python-openssl git ffmpeg libjpeg-dev

RUN mkdir -p /tmp/build && \
    cd /tmp/build && \
    wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz && \
    tar xzf Python-3.6.8.tgz && \
    cd Python-3.6.8 && \
    ./configure && \
    make && \
    make install && \
    rm -rf /tmp/build

# RUN pip3 install tensorflow-gpu==1.8.0
RUN pip3 install numpy

WORKDIR /code

#RUN pip3 install opencv-python-headless==4.5.1.48
# opencv-contrib-python==3.4.9.33

# RUN pip3 install opencv-python-headless==4.1.1.26 

COPY docker-requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# RUN pip3 install scipy==1.2.1

ARG INCUBATOR_VER=1
COPY dependencies/deepercut /dependencies/deepercut
WORKDIR /dependencies/deepercut
RUN pip3 install -e /dependencies/deepercut

COPY . /code

WORKDIR /code

RUN pip3 install -e /code

