import os, csv, base64, time, threading, logging
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory

# ─── КОНФИГУРАЦИЯ ────────────────────────────────────────────────────────────
TG_TOKEN   = "8665430525:AAEHC1Miy7cdiaOqXNW72vnooOn4u8A_T_o"
TG_CHAT_ID = "-5373500861"   # ID вашей группы/чата

FOTO_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Foto")
CSV_PATH   = os.path.join(FOTO_DIR, "spisok.csv")
MODEL_PATH = os.path.join(FOTO_DIR, "model.yml")
CASCADE    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
COOLDOWN   = 1800   # секунд между повторными уведомлениями
CONFIDENCE_THRESHOLD = 80   # порог LBPH (меньше = строже)

os.makedirs(FOTO_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─── ГЛОБАЛЬНОЕ СОСТОЯНИЕ ────────────────────────────────────────────────────
recognizer   = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(CASCADE)
label_map    = {}       # emp_id -> name
model_ready  = False
last_seen    = {}       # name -> unix timestamp
arrival_log  = []       # список приходов
lock         = threading.Lock()

# ─── CSV ─────────────────────────────────────────────────────────────────────
def load_csv():
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({"id": int(row["id"]), "name": row["name"]})
            except Exception:
                pass
    return rows

def save_csv(employees):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name"])
        w.writeheader()
        w.writerows(employees)

def next_id(employees):
    return max((e["id"] for e in employees), default=0) + 1

# ─── ОБУЧЕНИЕ МОДЕЛИ ─────────────────────────────────────────────────────────
def train_model():
    global recognizer, label_map, model_ready
    employees = load_csv()
    if not employees:
        with lock:
            model_ready = False
            label_map = {}
        log.info("No employees — model not trained")
        return False

    faces, labels, new_map = [], [], {}

    for emp in employees:
        emp_dir = os.path.join(FOTO_DIR, str(emp["id"]))
        if not os.path.isdir(emp_dir):
            continue
        imgs = [f for f in os.listdir(emp_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in imgs:
            img = cv2.imread(os.path.join(emp_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            detected = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(50, 50))
            if len(detected):
                x, y, w, h = detected[0]
                roi = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            else:
                roi = cv2.resize(img, (200, 200))
            faces.append(roi)
            labels.append(emp["id"])
            new_map[emp["id"]] = emp["name"]

    if not faces:
        log.warning("train_model: no usable face images found")
        with lock:
            model_ready = False
        return False

    new_rec = cv2.face.LBPHFaceRecognizer_create()
    new_rec.train(faces, np.array(labels))
    new_rec.save(MODEL_PATH)

    with lock:
        recognizer = new_rec
        label_map = new_map
        model_ready = True
    log.info(f"Model trained: {len(faces)} samples, {len(new_map)} employees")
    return True

def load_saved_model():
    global model_ready, label_map
    if not os.path.exists(MODEL_PATH):
        return
    emps = load_csv()
    if not emps:
        return
    try:
        recognizer.read(MODEL_PATH)
        label_map = {e["id"]: e["name"] for e in emps}
        model_ready = True
        log.info("Saved model loaded OK")
    except Exception as e:
        log.warning(f"Could not load model: {e}")

# ─── TELEGRAM ────────────────────────────────────────────────────────────────
def send_telegram(text: str, photo_path: str = None) -> bool:
    try:
        if photo_path and os.path.exists(photo_path):
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
            with open(photo_path, "rb") as ph:
                r = requests.post(
                    url,
                    data={"chat_id": TG_CHAT_ID, "caption": text, "parse_mode": "HTML"},
                    files={"photo": ph},
                    timeout=15
                )
        else:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            r = requests.post(
                url,
                json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=15
            )
        result = r.json()
        if not result.get("ok"):
            log.warning(f"TG error: {result.get('description')} | chat_id={TG_CHAT_ID}")
        return result.get("ok", False)
    except Exception as e:
        log.warning(f"TG exception: {e}")
        return False

# ─── РАСПОЗНАВАНИЕ ───────────────────────────────────────────────────────────
def recognize_frame(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    results = []
    for (x, y, w, h) in detected:
        roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        if model_ready and label_map:
            with lock:
                label_id, confidence = recognizer.predict(roi)
            if confidence < CONFIDENCE_THRESHOLD:
                name = label_map.get(label_id, "Неизвестно")
            else:
                name = "Неизвестно"
                label_id = -1
        else:
            name = "Нет модели"
            label_id = -1
            confidence = 999.0
        results.append({
            "name": name,
            "confidence": round(float(confidence), 1),
            "box": [int(x), int(y), int(w), int(h)],
            "id": label_id
        })
    return results

def handle_arrivals(results, frame_b64: str):
    now = time.time()
    for r in results:
        name = r["name"]
        if name in ("Неизвестно", "Нет модели"):
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

        snap_path = None
        try:
            raw = base64.b64decode(frame_b64)
            arr = np.frombuffer(raw, np.uint8)
            snap = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            snap_path = os.path.join(FOTO_DIR, f"snap_{name}_{int(now)}.jpg")
            cv2.imwrite(snap_path, snap)
        except Exception as e:
            log.warning(f"Snapshot error: {e}")

        tg_ok = send_telegram(text, snap_path)
        arrival_log.insert(0, {
            "name": name, "time": time_str, "date": date_str,
            "tg_ok": tg_ok, "ts": int(now)
        })
        if len(arrival_log) > 300:
            arrival_log.pop()
        log.info(f"Arrival: {name} | TG={'OK' if tg_ok else 'FAIL'}")

# ─── МАРШРУТЫ ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "model_ready": model_ready,
        "employees":   len(label_map),
        "log_count":   len(arrival_log),
        "tg_token":    bool(TG_TOKEN),
        "tg_chat":     TG_CHAT_ID,
    })

@app.route("/api/employees")
def api_employees():
    employees = load_csv()
    result = []
    for e in employees:
        emp_dir = os.path.join(FOTO_DIR, str(e["id"]))
        count = 0
        if os.path.isdir(emp_dir):
            count = len([f for f in os.listdir(emp_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        result.append({**e, "photo_count": count,
                        "thumb": f"/foto/{e['id']}/thumb" if count else None})
    return jsonify(result)

@app.route("/foto/<int:emp_id>/thumb")
def emp_thumb(emp_id):
    emp_dir = os.path.join(FOTO_DIR, str(emp_id))
    if os.path.isdir(emp_dir):
        imgs = sorted([f for f in os.listdir(emp_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if imgs:
            return send_from_directory(emp_dir, imgs[0])
    return ("", 404)

@app.route("/api/add_employee", methods=["POST"])
def api_add_employee():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    photos_b64 = data.get("photos") or []

    if not name:
        return jsonify({"ok": False, "error": "Имя не указано"}), 400
    if not photos_b64:
        return jsonify({"ok": False, "error": "Нет фотографий"}), 400

    employees = load_csv()
    if any(e["name"].lower() == name.lower() for e in employees):
        return jsonify({"ok": False, "error": f"Сотрудник «{name}» уже существует"}), 400

    emp_id  = next_id(employees)
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
            log.warning(f"Photo save error #{i}: {e}")

    if saved == 0:
        return jsonify({"ok": False, "error": "Не удалось сохранить фотографии"}), 500

    employees.append({"id": emp_id, "name": name})
    save_csv(employees)
    threading.Thread(target=train_model, daemon=True).start()
    log.info(f"Added employee: {name} (id={emp_id}, photos={saved})")
    return jsonify({"ok": True, "id": emp_id, "name": name, "photos": saved})

@app.route("/api/delete_employee", methods=["POST"])
def api_delete_employee():
    data = request.get_json(force=True, silent=True) or {}
    emp_id = data.get("id")
    employees = [e for e in load_csv() if e["id"] != emp_id]
    save_csv(employees)
    import shutil
    emp_dir = os.path.join(FOTO_DIR, str(emp_id))
    if os.path.isdir(emp_dir):
        shutil.rmtree(emp_dir)
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    data = request.get_json(force=True, silent=True) or {}
    b64  = data.get("frame", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        arr   = np.frombuffer(base64.b64decode(b64), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    results = recognize_frame(frame)

    # Аннотируем кадр
    for r in results:
        x, y, w, h = r["box"]
        known = r["name"] not in ("Неизвестно", "Нет модели")
        color = (40, 167, 69) if known else (220, 53, 69)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        label = f"{r['name']}  ({r['confidence']:.0f})"
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    ann_b64 = base64.b64encode(buf).decode()

    if results:
        threading.Thread(target=handle_arrivals, args=(results, b64), daemon=True).start()

    return jsonify({
        "ok": True,
        "results":    results,
        "annotated":  ann_b64,
        "model_ready": model_ready,
        "employees":  len(label_map)
    })

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({"ok": True, "message": "Переобучение запущено"})

@app.route("/api/log")
def api_log():
    return jsonify(arrival_log[:100])

@app.route("/api/test_telegram", methods=["POST"])
def api_test_telegram():
    ok = send_telegram(
        "🟢 <b>FaceAttend — тест подключения</b>\n"
        "✅ Уведомления о приходе сотрудников будут приходить сюда."
    )
    return jsonify({"ok": ok, "chat_id": TG_CHAT_ID})

# ─── ЗАПУСК ──────────────────────────────────────────────────────────────────
load_saved_model()
if not model_ready:
    threading.Thread(target=train_model, daemon=True).start()

if __name__ == "__main__":
    print("=" * 50)
    print("  FaceAttend запущен!")
    print("  Откройте: http://localhost:5000")
    print(f"  Telegram Chat ID: {TG_CHAT_ID}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
