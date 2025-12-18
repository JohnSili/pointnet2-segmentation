#!/bin/bash

# Скрипт для подготовки данных для загрузки в Colab

echo "=== Подготовка данных для Colab ==="

DATA_DIR="3011-20251217T195928Z-1-001"
OUTPUT_ZIP="data_for_colab.zip"

# Проверка наличия данных
if [ ! -d "$DATA_DIR" ]; then
    echo "Ошибка: Директория $DATA_DIR не найдена!"
    exit 1
fi

echo "Найденные данные:"
echo "  Директория: $DATA_DIR"
echo "  PLY файлов: $(find $DATA_DIR -name '*.ply' | wc -l)"

# Создаем архив
echo ""
echo "Создание архива..."
zip -r "$OUTPUT_ZIP" "$DATA_DIR" -q

if [ $? -eq 0 ]; then
    SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
    echo "✓ Архив создан: $OUTPUT_ZIP ($SIZE)"
    echo ""
    echo "Теперь загрузите этот файл в Colab:"
    echo "  1. Откройте Colab"
    echo "  2. Используйте: from google.colab import files; uploaded = files.upload()"
    echo "  3. Выберите файл: $OUTPUT_ZIP"
    echo "  4. Распакуйте: !unzip -q data_for_colab.zip"
    echo ""
    echo "Или загрузите на Google Drive и подключите в Colab"
else
    echo "Ошибка при создании архива"
    exit 1
fi

