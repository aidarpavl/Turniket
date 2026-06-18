@echo off
echo ===================================
echo  Установка зависимостей FaceAttend
echo ===================================
echo.
echo Шаг 1: Обновляем pip...
python -m pip install --upgrade pip

echo.
echo Шаг 2: Устанавливаем пакеты...
pip install flask opencv-contrib-python pillow requests numpy pandas

echo.
echo ===================================
echo  Готово! Запускаем приложение...
echo ===================================
python app.py
pause
