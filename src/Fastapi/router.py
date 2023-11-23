from fastapi import APIRouter, HTTPException, Path
from fastapi import Depends
from config import SessionLocal
from sqlalchemy.orm import Session
from schemas import RequestVehicle, Vehicle, Request, Response, RequestVehicle

import crud

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/create")
async def create_service(request: RequestVehicle, db: Session = Depends(get_db)):
    crud.create_vehicle(db=db, vehicle=request.parameter)
    return Response(status="Ok",
                    code="200",
                    message="Created successfully").dict(exclude_none=True)


@router.get("/")
async def get_vehicle(db: Session = Depends(get_db)):
    _vehicle = crud.get_vehicle(db=db)
    return Response(status="Ok", code="200", message="Success fetch all data", result=_vehicle)

@router.get("/count")
async def get_vehicle_count(db: Session = Depends(get_db)):
    count = crud.get_vehicle_count(db=db)
    return Response(status="Ok", code="200", message="Success", result=[count] )

@router.get("/{vehicle_id}")
async def get_vehicle_by_id(vehicle_id: int, db: Session = Depends(get_db)):
    _vehicle = crud.get_vehicle_by_id(db=db, vehicle_id=vehicle_id)
    if _vehicle is None:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return Response(status="Ok", code="200", message="Success", result=_vehicle)

@router.get("/color/{color}")
async def get_vehicle_by_color(color: str, db: Session = Depends(get_db)):
    _vehicle = crud.get_vehicles_by_color(db=db, color=color)
    if _vehicle is None:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return Response(status="Ok", code="200", message="Success", result=[_vehicle])

@router.get("/num_plate/{num_plate}")
async def get_vehicle_by_num_plate(num_plate: str, db: Session = Depends(get_db)):
    _vehicle = crud.get_vehicle_by_num_plate(db=db, num_plate=num_plate)
    if _vehicle is None:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return Response(status="Ok", code="200", message="Success", result=_vehicle)


#Get vehicle by color
#Get all vehicles
#Get vehicle by number plate