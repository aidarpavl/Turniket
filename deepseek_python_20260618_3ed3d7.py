import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
import asyncio
from PIL import Image
import threading
import base64
import io

# ================= ОБРАБОТКА ОШИБОК ИМПОРТА =================
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV (cv2) не установлен. Некоторые функции будут недоступны.")
    st.info("Для установки выполните: pip install opencv-python-headless")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    st.warning("⚠️ face_recognition не установлен. Используется упрощенный режим.")
    st.info("Для установки выполните: pip install face-recognition")

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    st.warning("⚠️ python-telegram-bot не установлен. Уведомления недоступны.")
    st.info("Для установки выполните: pip install python-telegram-bot")

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = "8665430525:AAEHC1Miy7cdiaOqXNW72vnooOn4u8A_T_o id"  # Ваш токен
TELEGRAM_CHAT_ID = "-5373500861"  # Ваш ID чата
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
    st.cache_data.clear()

# ================= УПРОЩЕННОЕ РАСПОЗНАВАНИЕ БЕЗ FACE_RECOGNITION =================
class SimpleFaceDetector:
    def __init__(self):
        self.known_names = []
        self.known_encodings = []
        self.recognized_today = set()
        self.last_recognition_time = {}
        self.last_detected = {}
        self.detection_cooldown = 5  # секунд между распознаваниями одного человека
        
    def load_known_faces(self):
        df = load_employees()
        self.known_names = df['name'].tolist() if 'name' in df.columns else []
        return self.known_names

    def detect_faces_simple(self, frame):
        """Упрощенная детекция без face_recognition"""
        if not CV2_AVAILABLE:
            return frame, []
        
        # Используем каскады Хаара для детекции лиц
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        self.load_known_faces()
        
        for (x, y, w, h) in faces:
            # Определяем, кто это (упрощенно - просто проверяем по имени)
            name = "Неизвестный"
            
            # Простая эвристика для определения знакомого лица
            if self.known_names:
                # Если есть сотрудники, проверяем, не похоже ли это лицо на кого-то
                # В упрощенном режиме просто используем первое имя для демонстрации
                if len(self.known_names) > 0:
                    # Для демонстрации используем имя первого сотрудника
                    # В реальном режиме нужно использовать face_recognition
                    name = self.known_names[0]
                    
                    # Отправляем уведомление, если не отправляли сегодня
                    current_time = time.time()
                    if name not in self.recognized_today:
                        self.recognized_today.add(name)
                        self.last_recognition_time[name] = current_time
                        # Отправляем уведомление в Telegram (если доступно)
                        if TELEGRAM_AVAILABLE:
                            send_message_sync(name)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        return frame, faces

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
        print(f"Ошибка отправки в Telegram: {e}")
        return False

def send_message_sync(name):
    try:
        if TELEGRAM_AVAILABLE:
            asyncio.run(send_telegram_message(name))
        return True
    except:
        return False

# ================= ОСНОВНОЙ КЛАСС ДЛЯ КАМЕРЫ =================
class FaceRecognitionApp:
    def __init__(self):
        self.cap = None
        self.detector = SimpleFaceDetector()
        self.use_advanced = FACE_RECOGNITION_AVAILABLE and CV2_AVAILABLE
        
    def start_camera(self):
        if not CV2_AVAILABLE:
            st.error("❌ OpenCV не доступен. Установите opencv-python-headless")
            return False
            
        if self.cap is None or not self.cap.isOpened():
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    # Пробуем использовать другую камеру
                    self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    # Пробуем использовать V4L2 (для Linux)
                    self.cap = cv2.VideoCapture('/dev/video0')
                    
                if not self.cap.isOpened():
                    st.error("❌ Не удалось открыть камеру. Проверьте подключение.")
                    return False
            except Exception as e:
                st.error(f"❌ Ошибка при открытии камеры: {e}")
                return False
        return True
    
    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            if self.use_advanced:
                # Расширенное распознавание с помощью face_recognition
                return self.process_frame_advanced(frame)
            else:
                # Упрощенное распознавание
                return self.detector.detect_faces_simple(frame)
                
        except Exception as e:
            st.warning(f"Ошибка обработки кадра: {e}")
            return frame if 'frame' in locals() else None
    
    def process_frame_advanced(self, frame):
        """Расширенное распознавание с использованием face_recognition"""
        if not FACE_RECOGNITION_AVAILABLE:
            return frame, []
        
        # Уменьшаем кадр для ускорения
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Обновляем данные сотрудников
        df = load_employees()
        known_encodings = []
        known_names = []
        
        for idx, row in df.iterrows():
            try:
                if os.path.exists(row['photo_path']):
                    image = face_recognition.load_image_file(row['photo_path'])
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(row['name'])
            except:
                continue
        
        # Находим лица
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        recognized_names = []
        
        for face_encoding in face_encodings:
            name = "Неизвестный"
            
            if known_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                    
                    # Проверяем, не отправляли ли уведомление сегодня
                    current_time = time.time()
                    if name not in self.detector.recognized_today:
                        self.detector.recognized_today.add(name)
                        self.detector.last_recognition_time[name] = current_time
                        if TELEGRAM_AVAILABLE:
                            send_message_sync(name)
            
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
    
    # Показываем статус библиотек
    with st.expander("📋 Статус компонентов"):
        st.write(f"✅ OpenCV: {'Доступен' if CV2_AVAILABLE else '❌ Не доступен'}")
        st.write(f"✅ Face Recognition: {'Доступен' if FACE_RECOGNITION_AVAILABLE else '❌ Не доступен'}")
        st.write(f"✅ Telegram Bot: {'Доступен' if TELEGRAM_AVAILABLE else '❌ Не доступен'}")
    
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
            elif not CV2_AVAILABLE:
                st.error("OpenCV не доступен!")
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
                    
                    if FACE_RECOGNITION_AVAILABLE:
                        # Кодируем лицо
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        if face_encodings:
                            encoding_str = str(face_encodings[0].tolist())
                            save_employee(new_name, photo_path, encoding_str)
                            st.success(f"✅ Сотрудник {new_name} успешно добавлен!")
                        else:
                            st.error("⚠️ Лицо не найдено на фото!")
                            os.remove(photo_path)
                    else:
                        # Упрощенное добавление без кодирования
                        save_employee(new_name, photo_path, None)
                        st.success(f"✅ Сотрудник {new_name} добавлен в упрощенном режиме!")
                    
                    st.session_state.photo_taken = True
                    st.cache_data.clear()
                    st.rerun()
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
        
        if st.session_state.camera_running and CV2_AVAILABLE:
            # Обрабатываем кадры в цикле
            while st.session_state.camera_running:
                frame = st.session_state.app.process_frame()
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB")
                else:
                    st.warning("Нет данных с камеры")
                    break
                
                time.sleep(0.05)
        elif not CV2_AVAILABLE:
            frame_placeholder.error("OpenCV не доступен. Установите opencv-python-headless")
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
        
        if hasattr(st.session_state.app, 'detector') and hasattr(st.session_state.app.detector, 'recognized_today'):
            recognized = st.session_state.app.detector.recognized_today
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