import os, csv, json, time
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────
TG_TOKEN   = "8665430525:AAEHC1Miy7cdiaOqXNW72vnooOn4u8A_T_o"
TG_CHAT_ID = "-5373500861"
COOLDOWN   = 1800
THRESHOLD  = 0.82

BASE_DIR = Path(__file__).parent
FOTO_DIR = BASE_DIR / "Foto"
CSV_PATH = FOTO_DIR / "spisok.csv"
EMB_PATH = FOTO_DIR / "embeddings.json"
FOTO_DIR.mkdir(exist_ok=True)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAttend — Турникет",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden}
.card{background:white;border-radius:12px;padding:16px 20px;margin-bottom:12px;border:1px solid rgba(0,0,0,.08)}
.srow{display:flex;align-items:center;gap:10px;padding:5px 0;font-size:13px}
.dg{width:10px;height:10px;border-radius:50%;background:#2d7d32;flex-shrink:0}
.dr{width:10px;height:10px;border-radius:50%;background:#c62828;flex-shrink:0}
.da{width:10px;height:10px;border-radius:50%;background:#e65100;flex-shrink:0}
.emp-row{display:flex;align-items:center;gap:10px;padding:8px 12px;background:#f7f8fa;border-radius:8px;margin-bottom:5px}
.ava{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0}
.log-row{display:flex;align-items:center;gap:10px;padding:10px 14px;background:#f7f8fa;border-radius:8px;margin-bottom:6px}
.lava{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0}
.bdg{font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px}
.bg{background:#e8f5e9;color:#2d7d32}
.bb{background:#e8f0fd;color:#1a6fc4}
.br{background:#ffebee;color:#c62828}
.ph-hdr{display:flex;align-items:center;gap:12px;padding:14px 18px;background:white;border-radius:12px;margin-bottom:14px;border:1px solid rgba(0,0,0,.08)}
.ph-icon{width:46px;height:46px;background:#1a6fc4;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0}
.sec{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#aaa;margin-bottom:6px}
</style>
""", unsafe_allow_html=True)

# ── ПАЛИТРА ЦВЕТОВ ────────────────────────────────────────────────────────────
PAL = [
    ("#c8e6c9","#1b5e20"),("#bbdefb","#0d47a1"),("#fce4ec","#880e4f"),
    ("#fff9c4","#f57f17"),("#ede7f6","#311b92"),("#fbe9e7","#bf360c"),
]
def pal(i): return PAL[i % len(PAL)]
def inits(n): parts=n.strip().split(); return "".join(p[0] for p in parts[:2]).upper() if parts else "?"

# ── FACE EMBEDDING (только PIL + numpy + skimage.feature) ────────────────────
def get_embedding(pil_img: Image.Image):
    try:
        gray = pil_img.convert("L").resize((128, 128))
        arr  = np.array(gray, dtype=np.float32) / 255.0
        mu, sigma = arr.mean(), arr.std()
        arr = arr + np.random.randn(*arr.shape) * 1e-4
        mu, sigma = arr.mean(), arr.std()
        arr = np.clip((arr - mu) / (sigma + 1e-6), -2, 2)
        arr = (arr + 2) / 4.0
        fd = hog(arr, orientations=9, pixels_per_cell=(8,8),
                 cells_per_block=(2,2), block_norm="L2-Hys")
        n = np.linalg.norm(fd)
        return fd / n if n > 1e-6 else None
    except Exception:
        return None

def face_crop(pil_img: Image.Image):
    w, h = pil_img.size
    side = int(min(w, h) * 0.85)
    x = (w - side) // 2
    y = (h - side) // 2
    return pil_img.crop((x, y, x+side, y+side))

def sim(a, b): return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0])

def recognize(pil_img: Image.Image):
    emb = get_embedding(face_crop(pil_img))
    if emb is None:
        return "Неизвестно", 0.0
    embs = st.session_state.embeddings
    employees = st.session_state.employees
    if not embs or not employees:
        return "Нет базы", 0.0
    best_name, best_score = "Неизвестно", 0.0
    thresh = st.session_state.get("threshold", THRESHOLD)
    for emp in employees:
        vlist = embs.get(emp["id"], [])
        if not vlist: continue
        score = max(sim(emb, np.array(v)) for v in vlist)
        if score > best_score:
            best_score = score
            best_name  = emp["name"] if score >= thresh else "Неизвестно"
    return best_name, best_score

def annotate(pil_img, name, score):
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    known = name not in ("Неизвестно", "Нет базы")
    col   = (40,167,69) if known else (220,53,69)
    side  = int(min(w,h)*0.85)
    x, y  = (w-side)//2, (h-side)//2
    draw.rectangle([x,y,x+side,y+side], outline=col, width=3)
    label = f"{name}  {int(score*100)}%" if known else name
    draw.rectangle([x, y-30, x+side, y], fill=col)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x+6, y-24), label, fill="white", font=font)
    return img

# ── CSV / EMBEDDINGS ──────────────────────────────────────────────────────────
def load_employees():
    if not CSV_PATH.exists(): return []
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try: rows.append({"id": int(row["id"]), "name": row["name"]})
            except: pass
    return rows

def save_employees(employees):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","name"])
        w.writeheader(); w.writerows(employees)

def next_id(employees):
    return max((e["id"] for e in employees), default=0) + 1

def load_embeddings():
    if not EMB_PATH.exists(): return {}
    try:
        with open(EMB_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): [np.array(v) for v in vl] for k,vl in raw.items()}
    except Exception: return {}

def save_embeddings(embs):
    with open(EMB_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k):[v.tolist() for v in vl] for k,vl in embs.items()}, f)

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_tg(text, photo=None):
    try:
        if photo:
            buf = BytesIO(); photo.save(buf,"JPEG",quality=85); buf.seek(0)
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                data={"chat_id":TG_CHAT_ID,"caption":text,"parse_mode":"HTML"},
                files={"photo":("snap.jpg",buf,"image/jpeg")}, timeout=15)
        else:
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id":TG_CHAT_ID,"text":text,"parse_mode":"HTML"}, timeout=15)
        return r.json().get("ok", False)
    except Exception: return False

def handle_arrival(name, photo):
    now = time.time()
    if now - st.session_state.last_seen.get(name, 0) < COOLDOWN:
        return False
    st.session_state.last_seen[name] = now
    dt = datetime.now()
    ts, ds = dt.strftime("%H:%M"), dt.strftime("%d.%m.%Y")
    text = f"✅ <b>Сотрудник прибыл</b>\n👤 <b>{name}</b>\n📅 {ds}  🕐 {ts}"
    tg_ok = send_tg(text, photo)
    st.session_state.log.insert(0, {"name":name,"time":ts,"date":ds,"tg_ok":tg_ok})
    return True

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "page"       not in st.session_state: st.session_state.page       = "camera"
if "log"        not in st.session_state: st.session_state.log        = []
if "last_seen"  not in st.session_state: st.session_state.last_seen  = {}
if "employees"  not in st.session_state: st.session_state.employees  = load_employees()
if "embeddings" not in st.session_state: st.session_state.embeddings = load_embeddings()
if "add_photos" not in st.session_state: st.session_state.add_photos = []
if "threshold"  not in st.session_state: st.session_state.threshold  = THRESHOLD

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:12px;
                border-bottom:1px solid #eee;margin-bottom:12px">
      <div style="width:40px;height:40px;background:#1a6fc4;border-radius:10px;
                  display:flex;align-items:center;justify-content:center;font-size:22px">👁</div>
      <div>
        <div style="font-size:15px;font-weight:700">FaceAttend</div>
        <div style="font-size:11px;color:#777">Турникет — учёт сотрудников</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Навигация
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📹 Камера", use_container_width=True,
                     type="primary" if st.session_state.page=="camera" else "secondary"):
            st.session_state.page = "camera"; st.rerun()
    with c2:
        if st.button("➕ Добавить", use_container_width=True,
                     type="primary" if st.session_state.page=="add" else "secondary"):
            st.session_state.page = "add"; st.rerun()

    st.divider()

    # Статус
    st.markdown('<div class="sec">Статус</div>', unsafe_allow_html=True)
    n_emp = len(st.session_state.employees)
    n_emb = len(st.session_state.embeddings)
    has_db = n_emp > 0 and n_emb > 0
    dm = "dg" if has_db else "dr"
    tm = f"База: {n_emp} сотрудников" if has_db else "База пуста — добавьте сотрудников"
    st.markdown(f"""
    <div class="srow"><div class="{dm}"></div><span>{tm}</span></div>
    <div class="srow"><div class="dg"></div><span>Telegram: {TG_CHAT_ID}</span></div>
    """, unsafe_allow_html=True)

    st.divider()

    # Список сотрудников
    st.markdown(f'<div class="sec">Сотрудники ({n_emp})</div>', unsafe_allow_html=True)
    if not st.session_state.employees:
        st.caption("Нет сотрудников. Нажмите «➕ Добавить».")
    else:
        for i, emp in enumerate(st.session_state.employees):
            bg, fg = pal(i)
            d = FOTO_DIR / str(emp["id"])
            cnt = len(list(d.glob("*.jpg"))) if d.is_dir() else 0
            cols = st.columns([6,1])
            with cols[0]:
                st.markdown(f"""
                <div class="emp-row">
                  <div class="ava" style="background:{bg};color:{fg}">{inits(emp['name'])}</div>
                  <div><div style="font-weight:600;font-size:13px">{emp['name']}</div>
                       <div style="font-size:11px;color:#888">{cnt} фото</div></div>
                </div>""", unsafe_allow_html=True)
            with cols[1]:
                if st.button("✕", key=f"d{emp['id']}"):
                    emps2 = [e for e in st.session_state.employees if e["id"]!=emp["id"]]
                    save_employees(emps2)
                    embs2 = {k:v for k,v in st.session_state.embeddings.items() if k!=emp["id"]}
                    save_embeddings(embs2)
                    emp_d = FOTO_DIR/str(emp["id"])
                    if emp_d.is_dir():
                        import shutil; shutil.rmtree(emp_d)
                    st.session_state.employees  = emps2
                    st.session_state.embeddings = embs2
                    st.rerun()

    st.divider()

    # Telegram тест
    if st.button("📨 Тест Telegram", use_container_width=True):
        ok = send_tg("🟢 <b>FaceAttend подключён!</b>\nСистема учёта сотрудников работает нормально.")
        st.success("✅ Отправлено!") if ok else st.error("❌ Ошибка. Бот должен быть администратором группы.")

    st.divider()
    st.markdown('<div class="sec">Настройки</div>', unsafe_allow_html=True)
    st.session_state.threshold = st.slider(
        "Порог распознавания", 0.60, 0.98, st.session_state.threshold, 0.01,
        help="0.82 рекомендуется. Выше = строже.")

# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: КАМЕРА
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "camera":
    st.markdown("""
    <div class="ph-hdr">
      <div class="ph-icon">📹</div>
      <div>
        <div style="font-size:18px;font-weight:700">Камера наблюдения</div>
        <div style="font-size:12px;color:#777">Сделайте снимок — система определит сотрудника и отправит уведомление в Telegram</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not has_db:
        st.warning("⚠️ База сотрудников пуста. Сначала добавьте сотрудников через «➕ Добавить».")

    col_cam, col_log = st.columns([3, 2])

    with col_cam:
        cam_img = st.camera_input(" ", label_visibility="collapsed")

        if cam_img:
            pil = Image.open(cam_img).convert("RGB")
            with st.spinner("Распознавание…"):
                name, score = recognize(pil)
                ann = annotate(pil, name, score)

            st.image(ann, use_container_width=True)

            known = name not in ("Неизвестно", "Нет базы")
            if known:
                st.success(f"✅ **{name}** — совпадение **{int(score*100)}%**")
                if handle_arrival(name, pil):
                    st.rerun()
            else:
                pct = int(score*100)
                st.error(f"❌ Не распознан (схожесть {pct}%)"
                         + (" — попробуйте снять чуть ближе" if pct > 60 else ""))

    with col_log:
        log = st.session_state.log
        st.markdown(f"**📋 Журнал прихода ({len(log)})**")

        if not log:
            st.info("Журнал пуст. Сделайте снимок.")
        else:
            if st.button("🗑 Очистить журнал"):
                st.session_state.log = []
                st.session_state.last_seen = {}
                st.rerun()
            for entry in log[:50]:
                idx = next((j for j,e in enumerate(st.session_state.employees)
                            if e["name"]==entry["name"]), -1)
                bg, fg = pal(idx) if idx>=0 else ("#e0e0e0","#555")
                tg = '<span class="bdg bb">✓ TG</span>' if entry["tg_ok"] else '<span class="bdg br">✗ TG</span>'
                st.markdown(f"""
                <div class="log-row">
                  <div class="lava" style="background:{bg};color:{fg}">{inits(entry['name'])}</div>
                  <div style="flex:1">
                    <div style="font-weight:600;font-size:13px">{entry['name']}</div>
                    <div style="font-size:11px;color:#777">{entry['date']} в {entry['time']}</div>
                  </div>
                  <div style="display:flex;gap:5px">
                    <span class="bdg bg">✓ Прибыл</span>{tg}
                  </div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: ДОБАВИТЬ СОТРУДНИКА
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "add":
    st.markdown("""
    <div class="ph-hdr">
      <div class="ph-icon">➕</div>
      <div>
        <div style="font-size:18px;font-weight:700">Добавить сотрудника</div>
        <div style="font-size:12px;color:#777">Введите имя и сделайте 3–5 фото с разных ракурсов</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1,1])

    with col_l:
        st.markdown("#### 👤 Имя сотрудника")
        emp_name = st.text_input("Имя и фамилия", placeholder="Иванов Айдар",
                                 key="inp_name", label_visibility="collapsed")

        st.markdown("#### 📷 Сделать снимки")
        st.caption("Сделайте 3–5 снимков: прямо, чуть влево, чуть вправо")

        snap = st.camera_input("Снимок", key="snap_cam", label_visibility="collapsed")
        if snap:
            pil = Image.open(snap).convert("RGB")
            st.session_state.add_photos.append(pil)
            st.success(f"📸 Снимок {len(st.session_state.add_photos)} добавлен!")

        cb1, cb2 = st.columns(2)
        with cb1:
            if st.button("🗑 Очистить снимки", use_container_width=True):
                st.session_state.add_photos = []; st.rerun()
        with cb2:
            ok_to_save = bool(emp_name.strip()) and len(st.session_state.add_photos) >= 1
            if st.button("✅ Сохранить", use_container_width=True,
                         type="primary", disabled=not ok_to_save):
                name = emp_name.strip()
                emps = st.session_state.employees
                if any(e["name"].lower()==name.lower() for e in emps):
                    st.error(f"Сотрудник «{name}» уже существует!")
                else:
                    emp_id  = next_id(emps)
                    emp_dir = FOTO_DIR / str(emp_id)
                    emp_dir.mkdir(exist_ok=True)
                    embs = st.session_state.embeddings
                    vlist = []
                    with st.spinner("Сохранение и обработка…"):
                        for idx, photo in enumerate(st.session_state.add_photos):
                            photo.save(str(emp_dir/f"{idx+1:03d}.jpg"), "JPEG", quality=90)
                            emb = get_embedding(face_crop(photo))
                            if emb is not None:
                                vlist.append(emb)
                    if not vlist:
                        st.error("❌ Не удалось извлечь данные лица. Сделайте фото ближе.")
                    else:
                        emps.append({"id": emp_id, "name": name})
                        save_employees(emps)
                        embs[emp_id] = vlist
                        save_embeddings(embs)
                        st.session_state.employees  = emps
                        st.session_state.embeddings = embs
                        st.session_state.add_photos = []
                        st.success(f"✅ Сотрудник **{name}** добавлен! ({len(vlist)} векторов)")
                        time.sleep(1.2)
                        st.session_state.page = "camera"; st.rerun()

    with col_r:
        st.markdown("#### 🖼 Сделанные снимки")
        photos = st.session_state.add_photos
        if not photos:
            st.info("Снимков пока нет. Используйте камеру слева.")
            st.markdown("""
            <div class="card">
              <div style="font-weight:600;margin-bottom:8px">💡 Советы</div>
              <div style="font-size:12px;color:#555;line-height:1.9">
                ✓ Хорошее освещение (лицо спереди)<br>
                ✓ Прямо в камеру<br>
                ✓ Снимок прямо + чуть влево + чуть вправо<br>
                ✓ Без очков и маски<br>
                ✓ Лицо занимает большую часть кадра
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.caption(f"Снимков: **{len(photos)}** (рекомендуется 3–5)")
            for row in range(0, len(photos), 2):
                c1, c2 = st.columns(2)
                with c1:
                    if row < len(photos):
                        st.image(photos[row], use_container_width=True)
                        if st.button("✕ Удалить", key=f"dp{row}"):
                            st.session_state.add_photos.pop(row); st.rerun()
                with c2:
                    if row+1 < len(photos):
                        st.image(photos[row+1], use_container_width=True)
                        if st.button("✕ Удалить", key=f"dp{row+1}"):
                            st.session_state.add_photos.pop(row+1); st.rerun()
