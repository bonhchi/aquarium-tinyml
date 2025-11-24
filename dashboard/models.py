from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# Bảng người dùng
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Bảng lưu trạng thái hệ thống (Lưu dữ liệu từ cảm biến gửi lên)
class SystemData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float, default=0.0)
    ph_level = db.Column(db.Float, default=7.0)
    light_status = db.Column(db.Boolean, default=False) # Tắt/Bật đèn
    pump_status = db.Column(db.Boolean, default=False)  # Tắt/Bật bơm
    timestamp = db.Column(db.DateTime, default=db.func.now())