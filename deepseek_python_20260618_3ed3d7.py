import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
import asyncio
from PIL import Image

# ================= ОБРАБОТКА ИМПОРТА TELEGRAM =================
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ python-telegram-bot не установлен")

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = "8665430525:AAEHC1Miy7cdiaOqXNW72vnooOn4u8A_T_o"
TELEGRAM_CHAT_ID = "-5373500861"
FOTO_DIR = "Foto"
CSV_FILE = "spisok.csv"

# Создаем папки
if not os.path.exists(FOTO_DIR):
    os.makedirs(FOTO_DIR)

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["name", "photo_path"])
    df.to_csv(CSV_FILE, index=False)

# ================= ЗАГРУЗКА ДАННЫХ =================
@st.cache_data
def load_employees():
    try:
        df = pd.read_csv(CSV_FILE)
        return df
    except:
        return pd.DataFrame(columns=["name", "photo_path"])

def save_employee(name, photo_path):
    df = load_employees()
    new_row = pd.DataFrame({
        "name": [name],
        "photo_path": [photo_path]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    st.cache_data.clear()

# ================= РАСПОЗНАВАНИЕ ЛИЦ (УПРОЩЕННОЕ) =================
def detect_faces_simple(frame):
    """Распознавание лиц с помощью каскадов Хаара"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces

# ================= ТЕЛЕГРАМ БОТ =================
async def send_telegram_message(name):
    if not TELEGRAM_AVAILABLE:
        return False
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        message = f"✅ Сотрудник {name} прибыл в {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        return True
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def send_message_sync(name):
    try:
        if TELEGRAM_AVAILABLE:
            asyncio.run(send_telegram_message(name))
        return True
    except:
        return False

# ================= ОСНОВНОЙ КЛАСС =================
class FaceRecognitionApp:
    def __init__(self):
        self.cap = None
        self.recognized_today = set()
        
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                st.error("❌ Не удалось открыть камеру")
                return False
            return True
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            return False
    
    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Находим лица
        faces = detect_faces_simple(frame)
        
        # Загружаем список сотрудников
        df = load_employees()
        employee_names = df['name'].tolist() if 'name' in df.columns else []
        
        # Обрабатываем каждое найденное лицо
        for (x, y, w, h) in faces:
            # Определяем имя (упрощенно - берем первого сотрудника)
            name = "Неизвестный"
            
            if employee_names:
                # Для демонстрации берем первого сотрудника
                name = employee_names[0]
                
                # Отправляем уведомление если не отправляли сегодня
                if name not in self.recognized_today:
                    self.recognized_today.add(name)
                    if TELEGRAM_AVAILABLE:
                        send_message_sync(name)
                        st.success(f"✅ Уведомление отправлено: {name}")
            
            # Рисуем рамку
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

# ================= ОСНОВНОЙ ИНТЕРФЕЙС =================
def main():
    st.set_page_config(page_title="Система распознавания", layout="wide")
    
    st.title("🔐 Система распознавания сотрудников")
    st.markdown("---")
    
    # Показываем статус
    with st.expander("📋 Статус системы"):
        st.write(f"✅ Telegram: {'Доступен' if TELEGRAM_AVAILABLE else '❌ Не доступен'}")
        st.write(f"✅ OpenCV: Доступен")
        st.write(f"✅ Камера: {'Готова' if cv2.VideoCapture(0).isOpened() else 'Не доступна'}")
    
    # Инициализация
    if 'app' not in st.session_state:
        st.session_state.app = FaceRecognitionApp()
    
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    # Боковая панель
    with st.sidebar:
        st.header("📋 Управление")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Запустить", use_container_width=True):
                if st.session_state.app.start_camera():
                    st.session_state.camera_running = True
                    st.rerun()
        with col2:
            if st.button("⏹️ Остановить", use_container_width=True):
                st.session_state.app.release_camera()
                st.session_state.camera_running = False
                st.rerun()
        
        st.markdown("---")
        
        # Добавление сотрудника
        st.subheader("➕ Добавить сотрудника")
        new_name = st.text_input("Имя")
        
        if st.button("📸 Сфотографировать", use_container_width=True):
            if not new_name:
                st.error("Введите имя!")
            else:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    photo_path = os.path.join(FOTO_DIR, f"{new_name}_{timestamp}.jpg")
                    cv2.imwrite(photo_path, frame)
                    save_employee(new_name, photo_path)
                    st.success(f"✅ {new_name} добавлен!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("❌ Ошибка фото!")
        
        st.markdown("---")
        
        # Список сотрудников
        st.subheader("👥 Сотрудники")
        df = load_employees()
        if len(df) > 0:
            st.dataframe(df[['name']], use_container_width=True)
            st.caption(f"Всего: {len(df)}")
        else:
            st.info("Нет сотрудников")
    
    # Основная область
    frame_placeholder = st.empty()
    
    if st.session_state.camera_running:
        while st.session_state.camera_running:
            frame = st.session_state.app.process_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
            time.sleep(0.05)
    else:
        frame_placeholder.info("Камера не запущена")

if __name__ == "__main__":
    main()