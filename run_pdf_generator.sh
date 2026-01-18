#!/bin/bash
# Скрипт для запуска PDF Generator с автоматической активацией виртуального окружения

echo "Активация виртуального окружения..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Ошибка активации виртуального окружения"
    exit 1
fi

echo "Запуск PDF Generator..."
# Подавляем GLib предупреждения
export GIO_USE_VFS=local
export GIO_MODULE_DIR=
python pdf_generator.py 2>/dev/null