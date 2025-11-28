from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Index,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import pytz

engine = create_engine("mysql+pymysql://nhom2:uitstudent@mysql.csc.edu.vn/SignalAPI")
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)
PredictionSessionLocal = SessionLocal


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


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pond_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, default=get_vietnam_time, nullable=False)
    temperature_c = Column(Float)
    ph = Column(Float)
    turbidity_ntu = Column(Float)
    water_level_cm = Column(Float)
    humidity_percent = Column(Float)
    raw_payload = Column(JSON)
    created_at = Column(DateTime, default=get_vietnam_time)

    __table_args__ = (Index("idx_pond_time", "pond_id", "timestamp"),)

Base.metadata.create_all(engine)
