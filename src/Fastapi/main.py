import asyncio
import pickle
import socket
from json import load

import crud
import cv2
import models
import numpy as np
import paho.mqtt.client as mqtt
from config import SessionLocal, engine
from fastapi import (Depends, FastAPI, HTTPException, Request, Response,
                     WebSocket)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from router import router
from webstream import router as rtsp_router
from webstream import router as webrouter

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
db = SessionLocal()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = mqtt.Client()
def on_connect(client, flags, rc, properties):
    client.subscribe("trafficflowyolov8")
    print("Connected: ", client, flags, rc, properties)

def message(client, userdata,payload):
    print("receiving")
    st=str(payload.payload.decode("utf-8"))
    s=list(map(str,st.split(",")))
    type=s[0]
    plate=s[1]
    color="red"
    print(s)
    vehicle = models.Vehicle_Base(title = str(type),num_plate=str(plate),color=str(color))
    with SessionLocal() as db: 
        db.add(vehicle)
        db.commit()
        db.refresh(vehicle)
    print("Received message: ", payload)

client.on_connect = on_connect
client.on_message = message

client.connect("broker.hivemq.com", 1883, 60)
# client.username_pw_set("myuser", "password")

@app.on_event("startup")
async def startup_event():
    client.loop_start()    

app.include_router(router, prefix="/vehicle", tags=["vehicles"])
app.include_router(webrouter, prefix="/web", tags=["web"])

host = "127.0.0.1"
port = 5001
max_length = 65000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

frame_info = None
buffer = None
frame = None

# Maintain a list of active WebSocket connections
active_connections = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Add the new WebSocket connection to the list of active connections
    active_connections.append(websocket)

    try:
        while True:
            data, address = sock.recvfrom(max_length)

            if len(data) < 100:
                frame_info = pickle.loads(data)

                if frame_info:
                    nums_of_packs = frame_info["packs"]

                    for i in range(nums_of_packs):
                        data, address = sock.recvfrom(max_length)

                        if i == 0:
                            buffer = data
                        else:
                            buffer += data

                    frame = np.frombuffer(buffer, dtype=np.uint8)
                    frame = frame.reshape(frame.shape[0], 1)

                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    # frame = cv2.flip(frame, 1)

                    if frame is not None and isinstance(frame, np.ndarray):
                        # Convert frame to JPEG image to send it over WebSocket
                        _, jpeg_frame = cv2.imencode('.jpg', frame)

                        # Broadcast the frame to all connected clients
                        for connection in active_connections:
                            await connection.send_bytes(jpeg_frame.tobytes())
                            await asyncio.sleep(0)

    except Exception as e:
        # Remove the WebSocket connection from the list of active connections if an error occurs or the client disconnects
        active_connections.remove(websocket)
        print(f"WebSocket disconnected: {e}")


@app.get("/template", response_class=HTMLResponse)
async def send_template(request: Request):
    car_count = crud.get_car_count(db=db)
    bike_count = crud.get_bike_count(db=db)
    bus_count = crud.get_bus_count(db=db)
    total_count = crud.get_vehicle_count(db=db)
    return templates.TemplateResponse("index.html", {"request": request, "car": car_count, "bike": bike_count, "bus": bus_count, "total": total_count}) 

