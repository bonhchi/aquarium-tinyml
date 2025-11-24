from flask import Flask, render_template, redirect, url_for, request, jsonify
from models import db, User, SystemData
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey_cua_ban' # Đổi thành chuỗi ngẫu nhiên
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db.init_app(app)

# Cấu hình Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Tạo database lần đầu chạy
with app.app_context():
    db.create_all()
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password') 
        # Thực tế nên mã hóa password bằng werkzeug.security
        
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    # Lấy dữ liệu mới nhất để hiển thị
    latest_data = SystemData.query.order_by(SystemData.timestamp.desc()).first()
    if not latest_data:
        latest_data = SystemData(temperature=0, ph_level=7) # Giá trị mặc định
        
    return render_template('dashboard.html', data=latest_data, name=current_user.username)

@app.route('/api/update_sensors', methods=['POST'])
def update_sensors():
    data = request.json # Nhận JSON từ ESP32
    # Ví dụ: {"temp": 28.5, "ph": 7.2}
    
    new_record = SystemData(
        temperature=data.get('temp'),
        ph_level=data.get('ph')
    )
    db.session.add(new_record)
    db.session.commit()
    return jsonify({"status": "success"}), 200

@app.route('/api/control', methods=['POST'])
@login_required
def control_device():
    device = request.form.get('device') # 'pump' hoặc 'light'
    action = request.form.get('action') # 'on' hoặc 'off'
    
    # Logic lưu trạng thái vào DB hoặc gửi MQTT (tùy hệ thống)
    # Ở đây ta giả sử cập nhật vào bản ghi mới nhất
    latest = SystemData.query.order_by(SystemData.id.desc()).first()
    if latest:
        if device == 'pump':
            latest.pump_status = (action == 'on')
        elif device == 'light':
            latest.light_status = (action == 'on')
        db.session.commit()
        
    return jsonify({"status": "updated", "device": device, "state": action})

# API 4.3: Lấy dữ liệu mới nhất (Cho Frontend AJAX gọi để cập nhật thẻ)
@app.route('/api/get_latest', methods=['GET'])
@login_required
def get_latest_data():
    latest = SystemData.query.order_by(SystemData.timestamp.desc()).first()
    if latest:
        return jsonify({
            "temp": latest.temperature,
            "ph": latest.ph_level,
            "light": latest.light_status,
            "pump": latest.pump_status
        })
    # Trả về giá trị mặc định nếu DB rỗng
    return jsonify({"temp": 0, "ph": 0, "light": False, "pump": False})

# 4.5 API MỚI: Cung cấp dữ liệu lịch sử cho biểu đồ
@app.route('/api/get_chart_data')
@login_required
def get_chart_data():
    # 1. Lấy 20 bản ghi mới nhất (đã lọc lỗi timestamp None)
    records = SystemData.query.filter(SystemData.timestamp != None)\
                              .order_by(SystemData.timestamp.desc())\
                              .limit(20).all()
    records.reverse() # Đảo lại để xếp theo thời gian tăng dần

    # 2. Chuẩn bị dữ liệu cho Biểu đồ Đường (Line) & Cột (Bar)
    labels = []
    temperatures = []
    phs = []
    
    # 3. Biến đếm cho Biểu đồ Tròn (Pie) - Thống kê Bơm
    pump_on_count = 0
    pump_off_count = 0

    for rec in records:
        time_str = rec.timestamp.strftime('%H:%M:%S')
        labels.append(time_str)
        temperatures.append(rec.temperature)
        phs.append(rec.ph_level)
        
        # Đếm trạng thái bơm
        if rec.pump_status:
            pump_on_count += 1
        else:
            pump_off_count += 1

    # 4. Trả về JSON chứa tất cả dữ liệu
    return jsonify({
        "labels": labels,
        "temperatures": temperatures,
        "phs": phs,
        "pump_stats": [pump_on_count, pump_off_count] # Dữ liệu mới cho Pie Chart
    })

# ... (Phần code điều khiển /api/control) ...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')