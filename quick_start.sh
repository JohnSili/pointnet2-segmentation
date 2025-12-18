#!/bin/bash

# Быстрый старт обучения

echo "=== PointNet++ Quick Start ==="

# Активируем окружение
source venv/bin/activate

# Создаем директории
mkdir -p checkpoints logs visualizations

# Запускаем обучение
echo "Запуск обучения..."
python train.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --num_points 2048 \
    --batch_size 4 \
    --epochs 50 \
    --lr 0.001 \
    --num_classes 13 \
    --device cuda \
    --save_dir ./checkpoints \
    --log_dir ./logs

echo "Обучение завершено!"

