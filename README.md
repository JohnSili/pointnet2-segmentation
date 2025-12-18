# PointNet++ для семантической сегментации облака точек

Проект для обучения нейронной сети PointNet++ для семантической сегментации 3D-облаков точек на датасете S3DIS (Stanford 3D Indoor Spaces).

## Описание

Этот проект реализует архитектуру PointNet++ для задачи семантической сегментации, где каждой точке 3D-облака присваивается метка класса (например, "стена", "пол", "стол", "кресло" и т.д.).

### Особенности

- Полная реализация архитектуры PointNet++ с Set Abstraction и Feature Propagation слоями
- Поддержка датасета S3DIS с 13 классами
- Обучение с Adam оптимизатором и StepLR scheduler
- Логирование в TensorBoard
- Метрики: Accuracy, IoU (Intersection over Union), Precision, Recall, F1-score
- Визуализация результатов с помощью matplotlib и Open3D

## Требования

- Python 3.8+
- PyTorch 1.9.0+
- CUDA (опционально, для GPU)

## Установка

1. Клонируйте репозиторий или скопируйте файлы проекта

2. Создайте виртуальное окружение (рекомендуется):
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

**Примечание:** Open3D опциональна (не поддерживает Python 3.13), но код работает без неё.

## Структура проекта

```
GDEM/
├── dataset.py          # Класс Dataset для загрузки данных S3DIS
├── pointnet2.py        # Архитектура PointNet++
├── train.py            # Скрипт обучения
├── test.py             # Скрипт тестирования
├── metrics.py          # Функции для вычисления метрик
├── visualize.py        # Скрипт визуализации
├── requirements.txt    # Зависимости проекта
└── README.md           # Документация
```

## Подготовка данных

### S3DIS Dataset

1. Скачайте датасет S3DIS с официального сайта или получите ссылку от преподавателя
2. Распакуйте данные в директорию `./data/`
3. Структура должна быть следующей:
```
data/
└── Area_1/
    ├── room1.txt
    ├── room2.txt
    └── ...
```

### Формат данных

Каждый файл должен содержать точки в формате:
- `x, y, z, r, g, b, label` (7 колонок)
- или `x, y, z, label` (4 колонки)
- или только `x, y, z` (3 колонки, цвета будут сгенерированы случайно)

Если данные отсутствуют, скрипт автоматически создаст синтетические данные для тестирования.

## Использование

### Обучение модели

**Важно:** Убедитесь, что виртуальное окружение активировано:
```bash
source venv/bin/activate  # Linux/Mac
```

Затем запустите обучение:
```bash
python train.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --num_points 4096 \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.001 \
    --num_classes 13 \
    --device cuda
```

**Для ваших данных (PLY файлы):**
```bash
python train.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --num_points 2048 \
    --batch_size 4 \
    --epochs 50 \
    --num_classes 13
```

Параметры:
- `--data_dir`: путь к директории с данными
- `--area`: область для использования (Area_1, Area_2, и т.д.)
- `--num_points`: количество точек в каждом облаке (по умолчанию 4096)
- `--batch_size`: размер батча (начните с 8-16 в зависимости от GPU)
- `--epochs`: количество эпох (50-100)
- `--lr`: начальный learning rate (по умолчанию 0.001)
- `--num_classes`: количество классов (13 для S3DIS)
- `--device`: устройство (cuda или cpu)
- `--save_dir`: директория для сохранения чекпоинтов (по умолчанию ./checkpoints)
- `--log_dir`: директория для логов TensorBoard (по умолчанию ./logs)

### Просмотр обучения в TensorBoard

```bash
tensorboard --logdir ./logs
```

Откройте браузер и перейдите по адресу `http://localhost:6006`

### Тестирование модели

```bash
python test.py \
    --data_dir ./data \
    --area Area_1 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13
```

### Визуализация результатов

```bash
python visualize.py \
    --data_dir ./data \
    --area Area_1 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_samples 5 \
    --output_dir ./visualizations
```

Для использования Open3D:
```bash
python visualize.py \
    --checkpoint ./checkpoints/best_model.pth \
    --use_open3d \
    --output_dir ./visualizations
```

## Архитектура модели

PointNet++ состоит из двух основных компонентов:

1. **Encoder (Set Abstraction Layers)**: 
   - Иерархическая выборка точек (Farthest Point Sampling)
   - Группировка точек (Ball Query)
   - Извлечение локальных признаков с помощью мини-сети PointNet

2. **Decoder (Feature Propagation Layers)**:
   - Интерполяция признаков обратно на исходные точки
   - Комбинирование локальных и глобальных признаков
   - Финальная классификация

## Метрики

Проект вычисляет следующие метрики:

- **Accuracy**: общая точность классификации
- **IoU (Intersection over Union)**: для каждого класса и средний IoU
- **Precision**: точность для каждого класса
- **Recall**: полнота для каждого класса
- **F1-Score**: гармоническое среднее precision и recall

## Классы S3DIS

Датасет S3DIS содержит 13 классов:
1. ceiling (потолок)
2. floor (пол)
3. wall (стена)
4. beam (балка)
5. column (колонна)
6. window (окно)
7. door (дверь)
8. table (стол)
9. chair (кресло)
10. sofa (диван)
11. bookcase (книжная полка)
12. board (доска)
13. clutter (прочее)

## Настройка гиперпараметров

### Рекомендуемые значения:

- **Learning Rate**: 0.001 (начальное), уменьшается в 2 раза каждые 20 эпох
- **Batch Size**: 8-16 (зависит от размера GPU)
- **Number of Points**: 4096 (можно уменьшить до 2048 для экономии памяти)
- **Epochs**: 50-100

### Для улучшения результатов:

- Увеличьте количество эпох
- Используйте аугментацию данных (добавьте в `dataset.py`)
- Настройте веса классов для дисбаланса данных
- Попробуйте разные learning rates и schedulers

## Устранение неполадок

### Проблема: Out of Memory

- Уменьшите `batch_size` (например, до 4)
- Уменьшите `num_points` (например, до 2048)
- Используйте CPU вместо GPU

### Проблема: Данные не загружаются

- Проверьте путь к данным
- Убедитесь, что файлы имеют правильный формат
- Скрипт автоматически создаст синтетические данные, если файлы не найдены

### Проблема: Медленное обучение

- Используйте GPU (CUDA)
- Увеличьте `num_workers` в DataLoader (если достаточно памяти)
- Уменьшите `num_points` или `batch_size`

## Лицензия

Этот проект создан в образовательных целях для практического занятия по сегментации облака точек.

## Запуск в Google Colab

Проект можно запустить в Google Colab для использования бесплатного GPU:

1. Откройте `colab_setup.ipynb` в Google Colab
2. Или клонируйте репозиторий:
   ```python
   !git clone https://github.com/yourusername/pointnet2-segmentation.git
   %cd pointnet2-segmentation
   !pip install -r requirements.txt
   ```
3. Следуйте инструкциям в `COLAB.md`

Подробная инструкция: [COLAB.md](COLAB.md)

## Загрузка на GitHub

Инструкция по загрузке проекта на GitHub: [GIT_SETUP.md](GIT_SETUP.md)

## Ссылки

- [PointNet++ Paper](https://arxiv.org/pdf/1706.02413)
- [PointNet++ GitHub](https://github.com/charlesq34/pointnet2)
- [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html)

