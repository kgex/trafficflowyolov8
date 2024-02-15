FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt install git -y
RUN apt-get install -y mosquitto mosquitto-clients
RUN apt-get install -y python3-paho-mqtt
RUN git clone https://github.com/kgex/trafficflowyolov8.git

WORKDIR /trafficflowyolov8

RUN pip3 install -r requirements.txt
RUN gdown https://drive.google.com/uc?id=1_Lsve_tfnv5weRMnYE93K1aANWyoRk8E

EXPOSE 1883 

CMD python3 scripts/main.py --input input/IMG_7315 - input.MOV