from sqlalchemy.orm import Session
from models import Vehicle_Base
from schemas import Vehicle


def get_vehicle(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Vehicle_Base).offset(skip).limit(limit).all()


def get_vehicle_by_id(db: Session, vehicle_id: int):
    return db.query(Vehicle_Base).filter(Vehicle_Base.id == vehicle_id).first()

def get_vehicle_count(db: Session):
    return db.query(Vehicle_Base).count()

def get_vehicles_by_color(color: str, db: Session):
    return db.query(Vehicle_Base).filter(Vehicle_Base.color == color).all()

def get_vehicle_by_num_plate(num_plate: str, db: Session):
    return db.query(Vehicle_Base).filter(Vehicle_Base.num_plate == num_plate).all()

def get_car_count(db: Session):
    return db.query(Vehicle_Base).filter(Vehicle_Base.title == "Car").count()

def get_bike_count(db: Session):
    return db.query(Vehicle_Base).filter(Vehicle_Base.title == "Bike").count()

def get_bus_count(db: Session):
    return db.query(Vehicle_Base).filter(Vehicle_Base.title == "Bus").count()

# def create_vehicle(db: Session, vehicle: Vehicle):
#     with open("a.txt", "r+") as file:

#         for line in file:
#             _vehicle =Vehicle_Base(title=str(line))
#             db.add(_vehicle)
#         db.commit()
#         db.refresh(_vehicle)
#         file.truncate()
#     return _vehicle

