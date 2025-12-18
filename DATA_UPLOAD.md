# Инструкция по загрузке данных в Colab

## Ваши данные

- **Директория:** `3011-20251217T195928Z-1-001/3011/`
- **Файлов:** 500 PLY файлов
- **Размер:** ~85 MB

## Способ 1: Создать архив и загрузить в Colab

### Шаг 1: Создать архив локально

```bash
cd /home/danil/Documents/GDEM
./prepare_data_for_colab.sh
```

Или вручную:
```bash
zip -r data_for_colab.zip 3011-20251217T195928Z-1-001
```

### Шаг 2: Загрузить в Colab

В Colab выполните:
```python
from google.colab import files
import zipfile

# Загрузите архив
uploaded = files.upload()

# Распакуйте
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Данные распакованы!")
```

### Шаг 3: Запустить обучение

```python
!python train.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --num_points 2048 \
    --batch_size 8 \
    --epochs 50 \
    --device cuda
```

## Способ 2: Загрузить на Google Drive

### Шаг 1: Загрузить архив на Google Drive

1. Создайте архив: `./prepare_data_for_colab.sh`
2. Загрузите `data_for_colab.zip` на Google Drive

### Шаг 2: Подключить Drive в Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Скопировать данные
!cp /content/drive/MyDrive/data_for_colab.zip .
!unzip -q data_for_colab.zip
```

### Шаг 3: Запустить обучение

```python
!python train.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --num_points 2048 \
    --batch_size 8 \
    --epochs 50 \
    --device cuda
```

## Способ 3: Использовать готовый ноутбук

Ноутбук `colab_setup.ipynb` уже настроен для автоматической загрузки данных:

1. Откройте: https://colab.research.google.com/github/JohnSili/pointnet2-segmentation/blob/main/colab_setup.ipynb
2. Выполните ячейки по порядку
3. В ячейке "Загрузка данных" загрузите архив `data_for_colab.zip`

## Проверка данных

После загрузки проверьте:
```python
import os
data_dir = '3011-20251217T195928Z-1-001'
area = '3011'
ply_files = [f for f in os.listdir(os.path.join(data_dir, area)) if f.endswith('.ply')]
print(f"Найдено {len(ply_files)} PLY файлов")
```

## Параметры для ваших данных

Рекомендуемые параметры:
- `--data_dir 3011-20251217T195928Z-1-001`
- `--area 3011`
- `--num_points 2048` (или 4096 если есть память)
- `--batch_size 8` (или меньше если не хватает памяти)
- `--num_classes 13` (или проверьте реальное количество классов в данных)

## Определение количества классов

```python
from dataset import S3DISDataset
import torch

dataset = S3DISDataset('3011-20251217T195928Z-1-001', area='3011', split='train', num_points=2048)
points, labels = dataset[0]
unique_classes = sorted(torch.unique(labels).tolist())
print(f"Классы в данных: {unique_classes}")
print(f"Количество классов: {len(unique_classes)}")
```

Используйте это число в параметре `--num_classes`.

