@echo off
REM Скрипт для запуска PDF Generator с автоматической активацией виртуального окружения

echo Активация виртуального окружения...
call venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Ошибка активации виртуального окружения
    pause
    exit /b 1
)

echo Запуск PDF Generator...
REM Подавляем GLib предупреждения
set GIO_USE_VFS=local
set GIO_MODULE_DIR=
python pdf_generator.py 2>nul

pause