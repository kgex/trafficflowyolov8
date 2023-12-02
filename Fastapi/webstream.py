import crud
import cv2
from config import SessionLocal
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from imagezmq import ImageHub
from sqlalchemy.orm import Session
import asyncio
image_hub = ImageHub(open_port='tcp://127.0.0.1:5566',REQ_REP = False)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()

templates = Jinja2Templates(directory="templates")

@router.get('/video')
async def video_feed(request: Request,db:Session=Depends(get_db)):
    # car_count = crud.get_car_count(db=db)
    return templates.TemplateResponse("video_feed.html", {"request": request})



@router.get('/video_feed')

async def video_feed():
    async def gen():
        while True:
            
            _, image = image_hub.recv_image()
            image = cv2.imencode('.jpg', image)[1].tostring()
            yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+image+b'\r\n'
            await asyncio.sleep(0)
    response = StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    async def on_response_disconnect():
        # This function is called when the client disconnects from the stream.
        # You can use it to perform cleanup actions.
        print('Client disconnected, cleaning up...')
        image_hub.close()

    # response.on('disconnect', on_response_disconnect)

    return response
