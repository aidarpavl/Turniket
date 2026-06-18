import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
import asyncio
from telegram import Bot
from PIL import Image
import threading

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = "8665430525:AAEHC1Miy7cdiaOqXNW72vnooOn4u8A_T_o id"  # Замените на ваш токен
TELEGRAM_CHAT_ID = "-5373500861"       # Ваш ID чата
FOTO_DIR = "Foto"
CSV_FILE = "spisok.csv"

# Создаем папки и файл, если их нет
if not os.path.exists(FOTO_DIR):
    os.makedirs(FOTO_DIR)

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["name", "photo_path", "encoding"])
    df.to_csv(CSV_FILE, index=False)

# ================= ЗАГРУЗКА ДАННЫХ =================
@st.cache_data
def load_employees():
    try:
        df = pd.read_csv(CSV_FILE)
        # Если нет колонки 'encoding', создаем ее
        if 'encoding' not in df.columns:
            df['encoding'] = None
        return df
    except:
        return pd.DataFrame(columns=["name", "photo_path", "encoding"])

def save_employee(name, photo_path, encoding):
    df = load_employees()
    new_row = pd.DataFrame({
        "name": [name],
        "photo_path": [photo_path],
        "encoding": [encoding]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    # Очищаем кеш для перезагрузки
    st.cache_data.clear()

# ================= ЗАГРУЗКА И КОДИРОВАНИЕ ЛИЦ =================
def get_known_encodings():
    df = load_employees()
    known_encodings = []
    known_names = []
    
    for idx, row in df.iterrows():
        try:
            if pd.notna(row['encoding']) and row['encoding'] is not None:
                # Если encoding сохранен в строковом виде
                if isinstance(row['encoding'], str):
                    encoding = np.fromstring(row['encoding'].strip('[]'), sep=' ')
                else:
                    encoding = row['encoding']
                known_encodings.append(encoding)
                known_names.append(row['name'])
            else:
                # Пробуем загрузить фото, если encoding нет
                if os.path.exists(row['photo_path']):
                    image = face_recognition.load_image_file(row['photo_path'])
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(row['name'])
        except:
            continue
    return known_encodings, known_names

# ================= ТЕЛЕГРАМ БОТ =================
async def send_telegram_message(name):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    message = f"✅ Сотрудник {name} прибыл в {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        return True
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")
        return False

def send_message_sync(name):
    try:
        asyncio.run(send_telegram_message(name))
        return True
    except:
        return False

# ================= ОСНОВНОЙ КЛАСС ДЛЯ КАМЕРЫ =================
class FaceRecognitionApp:
    def __init__(self):
        self.cap = None
        self.known_names = []
        self.known_encodings = []
        self.recognized_today = set()
        self.last_recognition_time = {}
        
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                st.error("Не удалось открыть камеру. Проверьте подключение.")
                return False
        return True
    
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
        
        # Уменьшаем кадр для ускорения
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Обновляем данные сотрудников
        self.known_encodings, self.known_names = get_known_encodings()
        
        # Находим лица
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        recognized_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Неизвестный"
            
            if True in matches:
                match_index = matches.index(True)
                name = self.known_names[match_index]
                
                # Проверяем, не отправляли ли уведомление сегодня
                current_time = time.time()
                if name not in self.recognized_today:
                    self.recognized_today.add(name)
                    self.last_recognition_time[name] = current_time
                    # Отправляем уведомление в Telegram
                    if send_message_sync(name):
                        st.success(f"✅ Уведомление отправлено: {name} прибыл!")
                    else:
                        st.warning(f"⚠️ Ошибка отправки уведомления для {name}")
            
            recognized_names.append(name)
        
        # Обновляем кадр для отображения
        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if name != "Неизвестный" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

# ================= ОСНОВНОЙ ИНТЕРФЕЙС STREAMLIT =================
def main():
    st.set_page_config(page_title="Система распознавания сотрудников", layout="wide")
    
    st.title("🔐 Система распознавания сотрудников")
    st.markdown("---")
    
    # Инициализация состояния сессии
    if 'app' not in st.session_state:
        st.session_state.app = FaceRecognitionApp()
    
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    if 'photo_taken' not in st.session_state:
        st.session_state.photo_taken = False
    
    # Боковая панель с информацией
    with st.sidebar:
        st.header("📋 Управление")
        
        # Кнопки управления камерой
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Запустить камеру", use_container_width=True):
                if st.session_state.app.start_camera():
                    st.session_state.camera_running = True
                    st.rerun()
        with col2:
            if st.button("⏹️ Остановить камеру", use_container_width=True):
                st.session_state.app.release_camera()
                st.session_state.camera_running = False
                st.rerun()
        
        st.markdown("---")
        
        # Добавление нового сотрудника
        st.subheader("➕ Добавить сотрудника")
        new_name = st.text_input("Имя сотрудника")
        
        if st.button("📸 Сфотографировать и добавить", use_container_width=True):
            if not new_name:
                st.error("Введите имя сотрудника!")
            elif not st.session_state.camera_running:
                st.error("Сначала запустите камеру!")
            else:
                # Делаем снимок
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Сохраняем фото
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    photo_path = os.path.join(FOTO_DIR, f"{new_name}_{timestamp}.jpg")
                    cv2.imwrite(photo_path, frame)
                    
                    # Кодируем лицо
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    if face_encodings:
                        encoding_str = str(face_encodings[0].tolist())
                        save_employee(new_name, photo_path, encoding_str)
                        st.success(f"✅ Сотрудник {new_name} успешно добавлен!")
                        st.session_state.photo_taken = True
                        # Очищаем кеш распознавания
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("⚠️ Лицо не найдено на фото! Попробуйте еще раз.")
                        os.remove(photo_path)  # Удаляем неудачное фото
                else:
                    st.error("❌ Не удалось сделать снимок!")
        
        st.markdown("---")
        
        # Список сотрудников
        st.subheader("👥 Сотрудники")
        df = load_employees()
        if len(df) > 0:
            st.dataframe(df[['name']], use_container_width=True)
            st.caption(f"Всего: {len(df)} сотрудников")
        else:
            st.info("Нет добавленных сотрудников")
    
    # Основная область
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Видеопоток")
        
        # Видео placeholder
        frame_placeholder = st.empty()
        
        if st.session_state.camera_running:
            # Обрабатываем кадры в цикле
            while st.session_state.camera_running:
                frame = st.session_state.app.process_frame()
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB")
                else:
                    st.warning("Нет данных с камеры")
                    break
                
                # Небольшая задержка для снижения нагрузки
                time.sleep(0.05)
        else:
            frame_placeholder.info("Камера не запущена. Нажмите 'Запустить камеру'")
    
    with col2:
        st.subheader("📊 Статус")
        
        if st.session_state.camera_running:
            st.success("🟢 Камера активна")
        else:
            st.error("🔴 Камера остановлена")
        
        st.markdown("---")
        
        st.subheader("📝 Последние уведомления")
        
        # Отображаем недавние распознавания
        if hasattr(st.session_state.app, 'recognized_today'):
            recognized = st.session_state.app.recognized_today
            if recognized:
                for name in list(recognized)[-5:]:
                    st.write(f"✅ {name} - {datetime.now().strftime('%H:%M')}")
            else:
                st.info("Нет уведомлений за сегодня")
        
        st.markdown("---")
        
        # Информация о сотрудниках
        df = load_employees()
        st.metric("Всего сотрудников", len(df))

if __name__ == "__main__":
    main()