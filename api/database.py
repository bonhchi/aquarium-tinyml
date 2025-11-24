from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import pytz

# engine = create_engine("sqlite:///signal.db")
engine = create_engine("mysql+pymysql://nhom2:uitstudent@mysql.csc.edu.vn/SignalAPI")  # Sử dụng pymysql driver
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


def get_vietnam_time():
    """Lấy thời gian hiện tại theo timezone Asia/Ho_Chi_Minh"""
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    return datetime.now(vietnam_tz)


class Turbidity(Base):
    __tablename__ = "turbidity"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50))
    raw = Column(Integer)
    ntu = Column(Float)
    timestamp = Column(DateTime, default=get_vietnam_time)


class TemperatureHumidity(Base):
    __tablename__ = "temperature_humidity"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50))
    temperature = Column(Float)
    humidity = Column(Float)
    timestamp = Column(DateTime, default=get_vietnam_time)


class Water(Base):
    __tablename__ = "water"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50))
    value = Column(Integer)
    timestamp = Column(DateTime, default=get_vietnam_time)

Base.metadata.create_all(engine)