#!/bin/bash

# Скрипт для настройки окружения проекта

echo "=== Настройка окружения PointNet++ ==="

# Создаем виртуальное окружение если его нет
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активируем окружение
echo "Активация виртуального окружения..."
source venv/bin/activate

# Обновляем pip
echo "Обновление pip..."
pip install --upgrade pip

# Устанавливаем зависимости
echo "Установка зависимостей..."
pip install -r requirements.txt

echo ""
echo "✓ Окружение настроено!"
echo ""
echo "Для активации окружения в будущем используйте:"
echo "  source venv/bin/activate"
echo ""
echo "Для запуска обучения:"
echo "  python train.py --data_dir 3011-20251217T195928Z-1-001 --area 3011 --num_points 2048 --batch_size 4"

