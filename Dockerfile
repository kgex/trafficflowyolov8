FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install git -y
RUN apt-get install -y mosquitto mosquitto-clients
RUN apt-get install -y python3-paho-mqtt

# To install the Gstreamer
RUN apt install -y \
    libssl3 \
    libssl-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libyaml-cpp-dev \
    gstreamer1.0-gtk3 gstreamer1.0-qt5

ADD . /trafficflowyolov8
WORKDIR /trafficflowyolov8

RUN pip3 install -r requirements.txt

EXPOSE 1883 

CMD python3 scripts/main.py --input input/final_input.MOV 