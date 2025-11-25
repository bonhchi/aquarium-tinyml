import requests
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

# ============================================================
# HÀM HỖ TRỢ LẤY DỮ LIỆU TỪ API NGOÀI
# ============================================================
def fetch_external_api(url):
    try:
        # Gọi API với timeout 2 giây để không làm treo server nếu mạng lag
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Giả sử API trả về 1 danh sách, ta lấy phần tử cuối cùng (mới nhất)
            if isinstance(data, list) and len(data) > 0:
                return data[-1] 
            return data # Hoặc trả về chính nó nếu không phải list
    except Exception as e:
        print(f"Lỗi khi gọi {url}: {e}")
        return None
    return None

# ============================================================
# SỬA LẠI API GET LATEST (QUAN TRỌNG NHẤT)
# ============================================================
# ... (Các import cũ giữ nguyên) ...

# Biến toàn cục để lưu trạng thái bơm tự động (giả lập lưu trong RAM)
# Trong thực tế, bạn nên lưu vào Database nếu muốn nó nhớ sau khi restart server
auto_pump_status = False 

@app.route('/api/get_latest', methods=['GET'])
@login_required
def get_latest_data():
    global auto_pump_status

    # 1. API các nguồn
    url_turbidity = "http://nhungapi.laptrinhpython.net/api/turbidity/all"
    url_temp_hum = "http://nhungapi.laptrinhpython.net/api/temperature_humidity/all"
    url_water = "http://nhungapi.laptrinhpython.net/api/water/all"

    # 2. Gọi API
    raw_turbidity = fetch_external_api(url_turbidity)
    raw_temp_hum = fetch_external_api(url_temp_hum)
    raw_water = fetch_external_api(url_water)
    
    # --- DEBUG: In ra terminal để xem API Độ đục trả về cái gì ---
    print("--- DEBUG TURBIDITY ---")
    print(raw_turbidity) 
    # -----------------------------------------------------------

    # 3. Chuẩn bị kết quả (QUAN TRỌNG: Phải khai báo đủ các key mặc định)
    result = {
        "temp": 0,
        "hum": 0,
        "water": 0,
        "turbidity": 0,    # <--- Lỗi 'undefined' do thiếu dòng này nếu API lỗi
        "pump_auto": auto_pump_status
    }

    # 4. Gán dữ liệu (Mapping)
    if raw_temp_hum:
        result["temp"] = raw_temp_hum.get("temperature", 0)
        result["hum"] = raw_temp_hum.get("humidity", 0)
        
    if raw_water:
        result["water"] = raw_water.get("distance", raw_water.get("value", 0))
    
    # Xử lý riêng cho Độ đục (Kiểm tra kỹ các key có thể xảy ra)
    if raw_turbidity:
        # Dựa vào hình ảnh bạn gửi, key chứa dữ liệu là "raw"
        # Chúng ta dùng .get("raw", 0) để lấy nó
        val = raw_turbidity.get("raw", 0)
        result["turbidity"] = val

    # Logic tự động hóa bơm (Giữ nguyên)
    if result["water"] < 2000:
        auto_pump_status = True
    elif result["water"] >= 4096:
        auto_pump_status = False

    result["pump_auto"] = auto_pump_status
    
    return jsonify(result)

# 4.5 API MỚI: Cung cấp dữ liệu lịch sử cho biểu đồ
# ... (Phần import và code cũ giữ nguyên) ...

@app.route('/api/get_chart_data')
@login_required
def get_chart_data():
    # 1. Định nghĩa link lấy TOÀN BỘ dữ liệu lịch sử
    url_turbidity_all = "http://nhungapi.laptrinhpython.net/api/turbidity/all"
    url_temp_hum_all = "http://nhungapi.laptrinhpython.net/api/temperature_humidity/all"
    url_water_all = "http://nhungapi.laptrinhpython.net/api/water/all"

    # 2. Hàm phụ trợ để lấy list dữ liệu an toàn
    def fetch_list(url):
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            return []
        except:
            return []

    # 3. Lấy dữ liệu từ 3 nguồn
    list_turbidity = fetch_list(url_turbidity_all)
    list_temp_hum = fetch_list(url_temp_hum_all)
    list_water = fetch_list(url_water_all)

    # 4. Xử lý dữ liệu: Chỉ lấy 20 phần tử CUỐI CÙNG (Mới nhất)
    # Lưu ý: Các API này có thể có số lượng bản ghi khác nhau, ta lấy theo list_temp_hum làm chuẩn
    limit = 20
    
    # Cắt 20 phần tử cuối
    data_temp_hum = list_temp_hum[-limit:] 
    data_water = list_water[-limit:]
    data_turbidity = list_turbidity[-limit:]

    # 5. Chuẩn bị mảng để vẽ
    labels = []       # Trục hoành (Thời gian)
    temps = []        # Nhiệt độ
    hums = []         # Độ ẩm
    waters = []       # Mực nước
    turbidities = []  # Độ đục

    # Duyệt qua danh sách nhiệt độ để tạo khung thời gian
    for item in data_temp_hum:
        # Giả sử API trả về field 'created_at' hoặc 'time', nếu không ta dùng số thứ tự
        # Ở đây ta lấy giờ từ chuỗi thời gian nếu có, hoặc để trống
        time_str = item.get('created_at', '') # Bạn cần kiểm tra key thực tế của API
        # Nếu time_str dài quá, ta cắt bớt chỉ lấy giờ:phút:giây
        if len(time_str) > 10:
             time_str = time_str[11:19] 
        
        labels.append(time_str)
        temps.append(item.get('temperature', 0))
        hums.append(item.get('humidity', 0))

    # Xử lý riêng cho Mực nước (vì list có thể lệch nhau, ta chỉ map theo index)
    for item in data_water:
        waters.append(item.get('distance', item.get('value', 0)))
    
    # Xử lý riêng cho Độ đục (SỬA LẠI ĐOẠN NÀY)
    for item in data_turbidity:
        # Lấy giá trị từ key "raw"
        turbidities.append(item.get('raw', 0))

    # Trả về JSON
    return jsonify({
        "labels": labels,
        "temps": temps,
        "hums": hums,
        "waters": waters,
        "turbidities": turbidities
    })

# API DÀNH RIÊNG CHO SERVER GATEWAY
@app.route('/api/gateway/command', methods=['GET'])
def gateway_command():
    global auto_pump_status
    
    # Lấy lại dữ liệu mực nước mới nhất một lần nữa để đảm bảo tính thời gian thực
    # (Hoặc bạn có thể tối ưu bằng cách lưu cache biến current_water_level ở trên)
    url_water = "http://nhungapi.laptrinhpython.net/api/water/all"
    raw_water = fetch_external_api(url_water)
    
    current_level = 0
    if raw_water:
        current_level = raw_water.get("distance", raw_water.get("value", 0))

    # Chạy lại logic để chắc chắn (Redundant check)
    command = "KEEP" # Giữ nguyên
    
    if current_level < 2000:
        auto_pump_status = True
        command = "ON"
    elif current_level >= 4096:
        auto_pump_status = False
        command = "OFF"
    else:
        # Nếu nằm giữa, trạng thái phụ thuộc vào biến auto_pump_status đang lưu
        command = "ON" if auto_pump_status else "OFF"

    # Trả về JSON theo định dạng chuẩn để Gateway dễ parse
    return jsonify({
        "device": "water_pump",
        "command": command,       # "ON" hoặc "OFF"
        "is_active": auto_pump_status, # true/false
        "current_level": current_level,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')