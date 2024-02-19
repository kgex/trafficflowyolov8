FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt install libgl1-mesa-glx -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install git -y
RUN apt-get install -y mosquitto mosquitto-clients
RUN apt-get install -y python3-paho-mqtt

ADD . /trafficflowyolov8
WORKDIR /trafficflowyolov8

RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio

EXPOSE 1883 

CMD python3 scripts/main.py --input input/final_input.MOV 