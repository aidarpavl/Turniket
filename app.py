import os, csv, base64, time, threading, json, logging
from io import BytesIO
from datetime import datetime, date

import cv2
import numpy as np
from PIL import Image
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TG_TOKEN   = "8665430525:AAEHC1Miy7cdiaOqXNW72vnooOn4u8A_T_o"
TG_CHAT_ID = "8665430525"          # owner personal chat (use numeric id)
FOTO_DIR   = os.path.join(os.path.dirname(__file__), "Foto")
CSV_PATH   = os.path.join(FOTO_DIR, "spisok.csv")
MODEL_PATH = os.path.join(FOTO_DIR, "model.yml")
CASCADE    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
COOLDOWN   = 1800   # seconds between repeat notifications per employee

os.makedirs(FOTO_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─── STATE ───────────────────────────────────────────────────────────────────
recognizer   = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(CASCADE)
label_map    = {}   # id -> name
model_ready  = False
last_seen    = {}   # name -> timestamp
arrival_log  = []   # [{name, time, date, tg_ok}]
lock         = threading.Lock()

# ─── CSV helpers ─────────────────────────────────────────────────────────────
def load_csv():
    employees = []
    if not os.path.exists(CSV_PATH):
        return employees
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            employees.append({"id": int(row["id"]), "name": row["name"]})
    return employees

def save_csv(employees):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name"])
        w.writeheader()
        w.writerows(employees)

def next_id(employees):
    return max((e["id"] for e in employees), default=0) + 1

# ─── Model training ──────────────────────────────────────────────────────────
def train_model():
    global recognizer, label_map, model_ready
    employees = load_csv()
    if not employees:
        model_ready = False
        label_map = {}
        return False

    faces, labels = [], []
    new_label_map = {}

    for emp in employees:
        emp_dir = os.path.join(FOTO_DIR, str(emp["id"]))
        if not os.path.isdir(emp_dir):
            continue
        imgs = [f for f in os.listdir(emp_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fname in imgs:
            img_path = os.path.join(emp_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            detected = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(60,60))
            if len(detected) == 0:
                # use whole image resized
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(emp["id"])
                new_label_map[emp["id"]] = emp["name"]
            else:
                x,y,w,h = detected[0]
                face_img = cv2.resize(img[y:y+h, x:x+w], (200, 200))
                faces.append(face_img)
                labels.append(emp["id"])
                new_label_map[emp["id"]] = emp["name"]

    if not faces:
        model_ready = False
        return False

    with lock:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        label_map = new_label_map
        model_ready = True
    log.info(f"Model trained: {len(faces)} samples, {len(new_label_map)} employees")
    return True

def load_saved_model():
    global model_ready, label_map
    if not os.path.exists(MODEL_PATH):
        return
    employees = load_csv()
    label_map = {e["id"]: e["name"] for e in employees}
    try:
        recognizer.read(MODEL_PATH)
        if label_map:
            model_ready = True
            log.info("Saved model loaded")
    except Exception as e:
        log.warning(f"Could not load model: {e}")

# ─── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(text: str, photo_path: str = None) -> bool:
    try:
        if photo_path and os.path.exists(photo_path):
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
            with open(photo_path, "rb") as ph:
                r = requests.post(url, data={"chat_id": TG_CHAT_ID, "caption": text, "parse_mode": "HTML"},
                                  files={"photo": ph}, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        result = r.json()
        if not result.get("ok"):
            log.warning(f"TG error: {result.get('description')}")
        return result.get("ok", False)
    except Exception as e:
        log.warning(f"TG exception: {e}")
        return False

# ─── Face recognition from frame ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 80   # lower = stricter (LBPH distance)

def recognize_frame(img_bgr):
    """Returns list of {name, confidence, box} for each detected face."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    results = []
    for (x, y, w, h) in detected:
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        if model_ready and label_map:
            with lock:
                label_id, confidence = recognizer.predict(face_roi)
            if confidence < CONFIDENCE_THRESHOLD:
                name = label_map.get(label_id, "Неизвестно")
            else:
                name = "Неизвестно"
                label_id = -1
        else:
            name = "Модель не обучена"
            label_id = -1
            confidence = 999
        results.append({"name": name, "confidence": round(float(confidence), 1),
                         "box": [int(x), int(y), int(w), int(h)], "id": label_id})
    return results

def handle_arrivals(results, frame_jpg_b64):
    """Check recognized employees and send Telegram if new arrival."""
    now = time.time()
    for r in results:
        name = r["name"]
        if name in ("Неизвестно", "Модель не обучена"):
            continue
        if now - last_seen.get(name, 0) < COOLDOWN:
            continue
        last_seen[name] = now
        dt = datetime.now()
        time_str = dt.strftime("%H:%M")
        date_str = dt.strftime("%d.%m.%Y")

        text = (f"✅ <b>Сотрудник прибыл</b>\n"
                f"👤 <b>{name}</b>\n"
                f"📅 {date_str}  🕐 {time_str}")

        # Save snapshot
        snap_path = None
        try:
            img_data = base64.b64decode(frame_jpg_b64)
            img_arr = np.frombuffer(img_data, np.uint8)
            snap_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            snap_path = os.path.join(FOTO_DIR, f"snap_{name}_{int(now)}.jpg")
            cv2.imwrite(snap_path, snap_bgr)
        except Exception as e:
            log.warning(f"Snap save error: {e}")

        tg_ok = send_telegram(text, snap_path)
        entry = {"name": name, "time": time_str, "date": date_str,
                 "tg_ok": tg_ok, "ts": int(now)}
        arrival_log.insert(0, entry)
        if len(arrival_log) > 200:
            arrival_log.pop()
        log.info(f"Arrival: {name}, TG={'OK' if tg_ok else 'FAIL'}")

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/employees")
def api_employees():
    employees = load_csv()
    result = []
    for e in employees:
        emp_dir = os.path.join(FOTO_DIR, str(e["id"]))
        photos = []
        if os.path.isdir(emp_dir):
            photos = [f for f in os.listdir(emp_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
        result.append({**e, "photo_count": len(photos),
                        "thumb": f"/foto/{e['id']}/thumb" if photos else None})
    return jsonify(result)

@app.route("/foto/<int:emp_id>/thumb")
def emp_thumb(emp_id):
    emp_dir = os.path.join(FOTO_DIR, str(emp_id))
    if os.path.isdir(emp_dir):
        imgs = sorted([f for f in os.listdir(emp_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        if imgs:
            return send_from_directory(emp_dir, imgs[0])
    return ("", 404)

@app.route("/api/add_employee", methods=["POST"])
def api_add_employee():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    photos_b64 = data.get("photos", [])   # list of base64 jpeg
    if not name:
        return jsonify({"ok": False, "error": "Имя не указано"}), 400
    if not photos_b64:
        return jsonify({"ok": False, "error": "Нет фотографий"}), 400

    employees = load_csv()
    # check duplicate
    if any(e["name"].lower() == name.lower() for e in employees):
        return jsonify({"ok": False, "error": f"Сотрудник '{name}' уже существует"}), 400

    emp_id = next_id(employees)
    emp_dir = os.path.join(FOTO_DIR, str(emp_id))
    os.makedirs(emp_dir, exist_ok=True)

    saved = 0
    for i, b64 in enumerate(photos_b64):
        try:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_data = base64.b64decode(b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img.save(os.path.join(emp_dir, f"{i+1:03d}.jpg"), "JPEG", quality=90)
            saved += 1
        except Exception as e:
            log.warning(f"Photo save error: {e}")

    if saved == 0:
        return jsonify({"ok": False, "error": "Не удалось сохранить фотографии"}), 500

    employees.append({"id": emp_id, "name": name})
    save_csv(employees)
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"ok": True, "id": emp_id, "name": name, "photos": saved})

@app.route("/api/delete_employee", methods=["POST"])
def api_delete_employee():
    data = request.get_json()
    emp_id = data.get("id")
    employees = load_csv()
    employees = [e for e in employees if e["id"] != emp_id]
    save_csv(employees)
    # remove photos folder
    import shutil
    emp_dir = os.path.join(FOTO_DIR, str(emp_id))
    if os.path.isdir(emp_dir):
        shutil.rmtree(emp_dir)
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Receives a base64 JPEG frame, returns recognition results + annotated frame."""
    data = request.get_json()
    b64 = data.get("frame", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img_data = base64.b64decode(b64)
        img_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    results = recognize_frame(frame)

    # Draw annotations
    for r in results:
        x,y,w,h = r["box"]
        known = r["name"] not in ("Неизвестно", "Модель не обучена")
        color = (40, 167, 69) if known else (220, 53, 69)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        label = f"{r['name']} ({r['confidence']:.0f})"
        cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(buf).decode()

    # Handle arrivals in background
    if results:
        threading.Thread(target=handle_arrivals, args=(results, b64), daemon=True).start()

    return jsonify({"ok": True, "results": results, "annotated": annotated_b64,
                    "model_ready": model_ready, "employees": len(label_map)})

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"ok": True, "message": "Переобучение запущено"})

@app.route("/api/log")
def api_log():
    return jsonify(arrival_log[:100])

@app.route("/api/status")
def api_status():
    return jsonify({
        "model_ready": model_ready,
        "employees": len(label_map),
        "log_count": len(arrival_log),
        "tg_token": bool(TG_TOKEN),
        "tg_chat": TG_CHAT_ID,
    })

@app.route("/api/test_telegram", methods=["POST"])
def api_test_telegram():
    ok = send_telegram("🟢 <b>FaceAttend подключён!</b>\nСистема распознавания лиц готова к работе.")
    return jsonify({"ok": ok})

# ─── STARTUP ─────────────────────────────────────────────────────────────────
load_saved_model()
if not model_ready:
    threading.Thread(target=train_model, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
