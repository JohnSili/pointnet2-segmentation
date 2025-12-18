#!/bin/bash

# Пример скрипта для запуска обучения PointNet++

echo "=== PointNet++ Training Example ==="

# Создаем директории
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p visualizations

# Обучение
echo "Starting training..."
python train.py \
    --data_dir ./data \
    --area Area_1 \
    --num_points 4096 \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.001 \
    --num_classes 13 \
    --device cuda \
    --save_dir ./checkpoints \
    --log_dir ./logs

# Тестирование (после обучения)
echo "Testing model..."
python test.py \
    --data_dir ./data \
    --area Area_1 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13

# Визуализация
echo "Creating visualizations..."
python visualize.py \
    --data_dir ./data \
    --area Area_1 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_samples 5 \
    --output_dir ./visualizations

echo "Done!"

