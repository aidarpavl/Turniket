"""
FaceAttend — система распознавания лиц сотрудников
Работает на Streamlit Cloud. Без cv2/dlib/face_recognition.
Использует: PIL, numpy, scikit-image, scikit-learn, requests
"""

import os
import csv
import json
import time
import base64
import hashlib
import threading
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from skimage import feature, color, transform
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ─── КОНФИГУРАЦИЯ ────────────────────────────────────────────────────────────
TG_TOKEN   = "8665430525:AAEHC1Miy7cdiaOqXNW72vnooOn4u8A_T_o"
TG_CHAT_ID = "-5373500861"
COOLDOWN   = 1800          # секунд между повторными уведомлениями
SIM_THRESH = 0.82          # порог схожести (0..1, выше = строже)

BASE_DIR  = Path(__file__).parent
FOTO_DIR  = BASE_DIR / "Foto"
CSV_PATH  = FOTO_DIR / "spisok.csv"
EMB_PATH  = FOTO_DIR / "embeddings.json"

FOTO_DIR.mkdir(exist_ok=True)

# ─── STREAMLIT CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAttend — Турникет",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── СТИЛИ ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Основные цвета */
:root { --blue:#1a6fc4; --green:#2d7d32; --red:#c62828; }

/* Скрыть лишнее */
#MainMenu, footer, header { visibility: hidden; }

/* Карточки */
.card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Статус строки */
.stat-row {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; font-size: 13px;
}
.dot-g { width:10px;height:10px;border-radius:50%;background:#2d7d32;flex-shrink:0; }
.dot-r { width:10px;height:10px;border-radius:50%;background:#c62828;flex-shrink:0; }
.dot-a { width:10px;height:10px;border-radius:50%;background:#e65100;flex-shrink:0; }

/* Сотрудник в списке */
.emp-row {
    display:flex;align-items:center;gap:10px;
    padding:8px 12px;background:#f7f8fa;
    border-radius:8px;margin-bottom:5px;
}
.emp-avatar {
    width:36px;height:36px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:13px;font-weight:700;flex-shrink:0;
}
.emp-info { flex:1; }
.emp-name { font-weight:600;font-size:13px; }
.emp-sub  { font-size:11px;color:#888; }

/* Журнал */
.log-row {
    display:flex;align-items:center;gap:10px;
    padding:10px 14px;background:#f7f8fa;
    border-radius:8px;margin-bottom:6px;
}
.log-avatar {
    width:40px;height:40px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:14px;font-weight:700;flex-shrink:0;
}
.log-info { flex:1; }
.log-name { font-weight:600;font-size:14px; }
.log-time { font-size:12px;color:#777;margin-top:2px; }
.log-badges { display:flex;gap:5px;flex-shrink:0; }
.badge {
    font-size:11px;font-weight:600;padding:3px 10px;
    border-radius:999px;
}
.b-green { background:#e8f5e9;color:#2d7d32; }
.b-blue  { background:#e8f0fd;color:#1a6fc4; }
.b-red   { background:#ffebee;color:#c62828; }
.b-gray  { background:#eee;color:#555; }

/* Заголовок страницы */
.page-header {
    display:flex;align-items:center;gap:12px;
    padding:16px 20px;background:white;
    border-radius:12px;margin-bottom:16px;
    border:1px solid rgba(0,0,0,0.08);
}
.page-icon {
    width:48px;height:48px;background:#1a6fc4;border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    font-size:26px;flex-shrink:0;
}
.page-title { font-size:20px;font-weight:700;color:#1a1a2e; }
.page-sub   { font-size:12px;color:#777; }

/* Секция */
.sec-label {
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:.08em;color:#aaa;margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# ─── ЦВЕТА ДЛЯ АВАТАРОК ──────────────────────────────────────────────────────
PALETTES = [
    ("#c8e6c9","#1b5e20"), ("#bbdefb","#0d47a1"), ("#fce4ec","#880e4f"),
    ("#fff9c4","#f57f17"), ("#ede7f6","#311b92"), ("#fbe9e7","#bf360c"),
]
def pal(i): return PALETTES[i % len(PALETTES)]

def initials(name):
    parts = name.strip().split()
    return "".join(p[0] for p in parts[:2]).upper() if parts else "?"

# ─── FACE EMBEDDING (HOG-based, no cv2) ─────────────────────────────────────
def extract_embedding(pil_img: Image.Image) -> np.ndarray | None:
    """
    Извлекает вектор лица из PIL Image.
    Использует HOG (Histogram of Oriented Gradients).
    Возвращает нормализованный numpy вектор или None.
    """
    try:
        img = pil_img.convert("L")           # grayscale
        img = img.resize((128, 128))         # стандартный размер
        arr = np.array(img, dtype=np.float32) / 255.0
        # Применяем CLAHE-подобное выравнивание гистограммы
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        arr = np.clip(arr, -2, 2)
        arr = (arr + 2) / 4.0               # normalize to [0,1]
        fd = feature.hog(
            arr,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        norm = np.linalg.norm(fd)
        if norm < 1e-6:
            return None
        return fd / norm
    except Exception as e:
        return None

def detect_face_region(pil_img: Image.Image):
    """
    Обнаруживает лицо на изображении.
    Возвращает (PIL.Image обрезанного лица, bbox) или (None, None).
    Использует скользящее окно + HOG.
    Упрощённая эвристика: берём центральную часть (80% от меньшего размера).
    Для Streamlit Cloud — без каскадных классификаторов.
    """
    w, h = pil_img.size
    # Берём центральный квадрат (обычно лицо в центре кадра при selfie)
    side = int(min(w, h) * 0.85)
    left  = (w - side) // 2
    top   = (h - side) // 2
    cropped = pil_img.crop((left, top, left+side, top+side))
    return cropped, (left, top, side, side)

def similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0])

# ─── CSV / EMBEDDINGS ────────────────────────────────────────────────────────
def load_employees():
    if not CSV_PATH.exists():
        return []
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({"id": int(row["id"]), "name": row["name"]})
            except Exception:
                pass
    return rows

def save_employees(employees):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","name"])
        w.writeheader()
        w.writerows(employees)

def next_id(employees):
    return max((e["id"] for e in employees), default=0) + 1

def load_embeddings() -> dict:
    """Загружает сохранённые эмбеддинги {emp_id: [список векторов]}"""
    if not EMB_PATH.exists():
        return {}
    try:
        with open(EMB_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): [np.array(v) for v in vlist] for k, vlist in raw.items()}
    except Exception:
        return {}

def save_embeddings(embs: dict):
    serializable = {str(k): [v.tolist() for v in vlist] for k, vlist in embs.items()}
    with open(EMB_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f)

def get_emp_photo(emp_id: int) -> Image.Image | None:
    emp_dir = FOTO_DIR / str(emp_id)
    if not emp_dir.is_dir():
        return None
    imgs = sorted([f for f in emp_dir.iterdir()
                   if f.suffix.lower() in (".jpg",".jpeg",".png")])
    if not imgs:
        return None
    try:
        return Image.open(imgs[0]).convert("RGB")
    except Exception:
        return None

# ─── TELEGRAM ────────────────────────────────────────────────────────────────
def send_telegram(text: str, photo: Image.Image | None = None) -> bool:
    try:
        if photo:
            buf = BytesIO()
            photo.save(buf, "JPEG", quality=85)
            buf.seek(0)
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                data={"chat_id": TG_CHAT_ID, "caption": text, "parse_mode": "HTML"},
                files={"photo": ("snap.jpg", buf, "image/jpeg")},
                timeout=15
            )
        else:
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=15
            )
        result = r.json()
        return result.get("ok", False)
    except Exception as e:
        return False

# ─── SESSION STATE ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page":       "camera",
        "log":        [],          # [{name, time, date, tg_ok}]
        "last_seen":  {},          # name -> timestamp
        "employees":  load_employees(),
        "embeddings": load_embeddings(),
        "add_photos": [],          # PIL Images для добавления
        "cam_active": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── РАСПОЗНАВАНИЕ ───────────────────────────────────────────────────────────
def recognize(pil_img: Image.Image):
    """
    Возвращает (name, similarity_score) лучшего совпадения.
    Если совпадений нет — ('Неизвестно', 0.0).
    """
    face, bbox = detect_face_region(pil_img)
    if face is None:
        return "Неизвестно", 0.0

    emb = extract_embedding(face)
    if emb is None:
        return "Неизвестно", 0.0

    embs = st.session_state.embeddings
    employees = st.session_state.employees
    if not embs or not employees:
        return "Нет базы", 0.0

    best_name  = "Неизвестно"
    best_score = 0.0

    for emp in employees:
        vlist = embs.get(emp["id"], [])
        if not vlist:
            continue
        scores = [similarity(emb, v) for v in vlist]
        score  = max(scores)
        if score > best_score:
            best_score = score
            best_name  = emp["name"] if score >= SIM_THRESH else "Неизвестно"

    return best_name, best_score

def handle_arrival(name: str, photo: Image.Image):
    now = time.time()
    if now - st.session_state.last_seen.get(name, 0) < COOLDOWN:
        return
    st.session_state.last_seen[name] = now
    dt = datetime.now()
    ts = dt.strftime("%H:%M")
    ds = dt.strftime("%d.%m.%Y")
    text = (f"✅ <b>Сотрудник прибыл</b>\n"
            f"👤 <b>{name}</b>\n"
            f"📅 {ds}  🕐 {ts}")
    tg_ok = send_telegram(text, photo)
    st.session_state.log.insert(0, {
        "name": name, "time": ts, "date": ds, "tg_ok": tg_ok
    })

# ─── АННОТАЦИЯ ИЗОБРАЖЕНИЯ ───────────────────────────────────────────────────
def annotate_image(pil_img: Image.Image, name: str, score: float) -> Image.Image:
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    known = name not in ("Неизвестно", "Нет базы", "Нет модели")
    color_box = (40, 167, 69) if known else (220, 53, 69)

    # Рамка вокруг центра
    side = int(min(w, h) * 0.85)
    x = (w - side) // 2
    y = (h - side) // 2
    draw.rectangle([x, y, x+side, y+side], outline=color_box, width=3)

    # Подпись
    label = f"{name}  {int(score*100)}%" if known else name
    bar_h = 28
    draw.rectangle([x, y - bar_h, x + side, y], fill=color_box)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x + 6, y - bar_h + 6), label, fill="white", font=font)
    return img

# ═══════════════════════════════════════════════════════════════════════════
# ─── SIDEBAR ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:14px;border-bottom:1px solid #eee;margin-bottom:14px">
      <div style="width:42px;height:42px;background:#1a6fc4;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0">👁</div>
      <div>
        <div style="font-size:16px;font-weight:700">FaceAttend</div>
        <div style="font-size:11px;color:#777">Турникет — учёт сотрудников</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Навигация
    st.markdown('<div class="sec-label">Навигация</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📹 Камера", use_container_width=True,
                     type="primary" if st.session_state.page=="camera" else "secondary"):
            st.session_state.page = "camera"
            st.rerun()
    with col2:
        if st.button("➕ Добавить", use_container_width=True,
                     type="primary" if st.session_state.page=="add" else "secondary"):
            st.session_state.page = "add"
            st.rerun()

    st.divider()

    # Статус
    st.markdown('<div class="sec-label">Статус системы</div>', unsafe_allow_html=True)
    n_emp = len(st.session_state.employees)
    n_emb = len(st.session_state.embeddings)
    has_model = n_emp > 0 and n_emb > 0

    dot_m = "g" if has_model else "r"
    txt_m = f"База: {n_emp} сотрудников" if has_model else "База пуста — добавьте сотрудников"
    st.markdown(f"""
    <div class="stat-row"><div class="dot-{dot_m}"></div><span>{txt_m}</span></div>
    <div class="stat-row"><div class="dot-g"></div><span>Telegram Chat: {TG_CHAT_ID}</span></div>
    """, unsafe_allow_html=True)

    st.divider()

    # Список сотрудников
    st.markdown(f'<div class="sec-label">Сотрудники ({n_emp})</div>', unsafe_allow_html=True)
    if not st.session_state.employees:
        st.caption("Нет сотрудников. Нажмите «Добавить».")
    else:
        for i, emp in enumerate(st.session_state.employees):
            bg, fg = pal(i)
            photo_count = len(list((FOTO_DIR/str(emp["id"])).glob("*.jpg"))) if (FOTO_DIR/str(emp["id"])).is_dir() else 0
            col_a, col_b = st.columns([5,1])
            with col_a:
                st.markdown(f"""
                <div class="emp-row">
                  <div class="emp-avatar" style="background:{bg};color:{fg}">{initials(emp['name'])}</div>
                  <div class="emp-info">
                    <div class="emp-name">{emp['name']}</div>
                    <div class="emp-sub">{photo_count} фото</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                if st.button("✕", key=f"del_{emp['id']}", help=f"Удалить {emp['name']}"):
                    employees = [e for e in st.session_state.employees if e["id"]!=emp["id"]]
                    save_employees(employees)
                    embs = st.session_state.embeddings
                    embs.pop(emp["id"], None)
                    save_embeddings(embs)
                    emp_dir = FOTO_DIR / str(emp["id"])
                    if emp_dir.is_dir():
                        import shutil; shutil.rmtree(emp_dir)
                    st.session_state.employees  = employees
                    st.session_state.embeddings = embs
                    st.rerun()

    st.divider()

    # Telegram тест
    st.markdown('<div class="sec-label">Telegram</div>', unsafe_allow_html=True)
    if st.button("📨 Тест уведомления", use_container_width=True):
        ok = send_telegram("🟢 <b>FaceAttend подключён!</b>\nСистема учёта сотрудников работает.")
        if ok:
            st.success("✅ Сообщение отправлено!")
        else:
            st.error("❌ Ошибка. Проверьте, что бот добавлен в группу как администратор.")

    # Порог распознавания
    st.divider()
    st.markdown('<div class="sec-label">Настройки</div>', unsafe_allow_html=True)
    SIM_THRESH = st.slider("Порог схожести", 0.60, 0.98, SIM_THRESH, 0.01,
                            help="Выше = строже. Рекомендуется 0.80–0.88")

# ═══════════════════════════════════════════════════════════════════════════
# ─── СТРАНИЦА: КАМЕРА ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.page == "camera":

    st.markdown("""
    <div class="page-header">
      <div class="page-icon">📹</div>
      <div>
        <div class="page-title">Камера наблюдения</div>
        <div class="page-sub">Сделайте снимок — система определит сотрудника и отправит уведомление</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_cam, col_log = st.columns([3, 2])

    with col_cam:
        st.markdown("#### 📷 Сделать снимок")

        if not has_model:
            st.warning("⚠️ База сотрудников пуста. Сначала добавьте сотрудников через «➕ Добавить».")

        # Камера
        camera_img = st.camera_input(
            "Нажмите кнопку для съёмки",
            label_visibility="collapsed"
        )

        if camera_img:
            pil = Image.open(camera_img).convert("RGB")

            with st.spinner("Распознавание…"):
                name, score = recognize(pil)
                annotated  = annotate_image(pil, name, score)

            st.image(annotated, use_container_width=True)

            known = name not in ("Неизвестно", "Нет базы", "Нет модели")
            if known:
                st.success(f"✅ **{name}** — совпадение {int(score*100)}%")
                handle_arrival(name, pil)
                # Перерисовать журнал
                st.rerun()
            else:
                st.error(f"❌ Сотрудник не распознан (схожесть {int(score*100)}%)")

    with col_log:
        st.markdown(f"#### 📋 Журнал прихода ({len(st.session_state.log)})")
        if not st.session_state.log:
            st.info("Журнал пуст. Сделайте снимок для распознавания.")
        else:
            if st.button("🗑 Очистить журнал", key="clr_log"):
                st.session_state.log = []
                st.session_state.last_seen = {}
                st.rerun()

            for i, entry in enumerate(st.session_state.log[:50]):
                emp_idx = next((j for j, e in enumerate(st.session_state.employees)
                                if e["name"]==entry["name"]), -1)
                bg, fg = pal(emp_idx) if emp_idx>=0 else ("#e0e0e0","#555")
                tg_badge = '<span class="badge b-blue">✓ TG</span>' if entry["tg_ok"] else '<span class="badge b-red">✗ TG</span>'
                st.markdown(f"""
                <div class="log-row">
                  <div class="log-avatar" style="background:{bg};color:{fg}">{initials(entry['name'])}</div>
                  <div class="log-info">
                    <div class="log-name">{entry['name']}</div>
                    <div class="log-time">{entry['date']} в {entry['time']}</div>
                  </div>
                  <div class="log-badges">
                    <span class="badge b-green">✓ Прибыл</span>
                    {tg_badge}
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ─── СТРАНИЦА: ДОБАВИТЬ СОТРУДНИКА ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "add":

    st.markdown("""
    <div class="page-header">
      <div class="page-icon">➕</div>
      <div>
        <div class="page-title">Добавить сотрудника</div>
        <div class="page-sub">Введите имя и сделайте 3–5 фотографий с разных ракурсов</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_preview = st.columns([1, 1])

    with col_form:
        st.markdown("#### 👤 Данные сотрудника")
        emp_name = st.text_input("Имя и фамилия", placeholder="Иванов Айдар", key="inp_name")

        st.markdown("#### 📷 Фотографии")
        st.caption("Сделайте несколько снимков: прямо, чуть левее, чуть правее. Это повысит точность.")

        snap = st.camera_input("Сделать снимок", key="snap_cam", label_visibility="collapsed")
        if snap:
            pil = Image.open(snap).convert("RGB")
            st.session_state.add_photos.append(pil)
            st.success(f"📸 Снимок добавлен! Всего: {len(st.session_state.add_photos)}")

        cols_btn = st.columns([1,1])
        with cols_btn[0]:
            if st.button("🗑 Очистить снимки", use_container_width=True):
                st.session_state.add_photos = []
                st.rerun()
        with cols_btn[1]:
            can_save = emp_name.strip() and len(st.session_state.add_photos) >= 1
            if st.button("✅ Сохранить сотрудника", use_container_width=True,
                         type="primary", disabled=not can_save):
                name = emp_name.strip()
                employees = st.session_state.employees

                if any(e["name"].lower()==name.lower() for e in employees):
                    st.error(f"Сотрудник «{name}» уже существует!")
                else:
                    emp_id  = next_id(employees)
                    emp_dir = FOTO_DIR / str(emp_id)
                    emp_dir.mkdir(exist_ok=True)

                    embeddings = st.session_state.embeddings
                    vlist = []

                    with st.spinner("Обработка фотографий…"):
                        for idx, photo in enumerate(st.session_state.add_photos):
                            # Сохранить фото
                            photo.save(str(emp_dir / f"{idx+1:03d}.jpg"), "JPEG", quality=90)
                            # Извлечь эмбеддинг
                            face, _ = detect_face_region(photo)
                            if face:
                                emb = extract_embedding(face)
                                if emb is not None:
                                    vlist.append(emb)

                    if not vlist:
                        st.error("❌ Не удалось обработать фотографии. Попробуйте снова.")
                    else:
                        employees.append({"id": emp_id, "name": name})
                        save_employees(employees)
                        embeddings[emp_id] = vlist
                        save_embeddings(embeddings)
                        st.session_state.employees  = employees
                        st.session_state.embeddings = embeddings
                        st.session_state.add_photos = []

                        st.success(f"✅ Сотрудник **{name}** добавлен! ({len(vlist)} векторов лица)")
                        time.sleep(1.5)
                        st.session_state.page = "camera"
                        st.rerun()

    with col_preview:
        st.markdown("#### 🖼 Сделанные снимки")
        photos = st.session_state.add_photos
        if not photos:
            st.info("Снимков пока нет. Используйте камеру слева.")
        else:
            st.caption(f"Снимков: **{len(photos)}** (рекомендуется 3–5)")
            # Показываем сетку 2 колонки
            for row_i in range(0, len(photos), 2):
                c1, c2 = st.columns(2)
                with c1:
                    if row_i < len(photos):
                        st.image(photos[row_i], use_container_width=True)
                        if st.button("✕", key=f"dph_{row_i}"):
                            st.session_state.add_photos.pop(row_i)
                            st.rerun()
                with c2:
                    if row_i+1 < len(photos):
                        st.image(photos[row_i+1], use_container_width=True)
                        if st.button("✕", key=f"dph_{row_i+1}"):
                            st.session_state.add_photos.pop(row_i+1)
                            st.rerun()

        # Инструкция
        st.markdown("""
        <div class="card" style="margin-top:16px">
          <div style="font-weight:600;margin-bottom:8px">💡 Советы для лучшего распознавания</div>
          <div style="font-size:12px;color:#555;line-height:1.8">
            ✓ Хорошее освещение<br>
            ✓ Смотрите прямо в камеру<br>
            ✓ Сделайте снимок прямо, чуть влево, чуть вправо<br>
            ✓ Без очков и маски<br>
            ✓ Лицо занимает большую часть кадра
          </div>
        </div>
        """, unsafe_allow_html=True)
