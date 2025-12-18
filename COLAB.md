# Запуск в Google Colab

## Быстрый старт

1. **Откройте ноутбук в Colab:**
   - Загрузите файл `colab_setup.ipynb` в Google Colab
   - Или скопируйте код из ноутбука в новый Colab notebook

2. **Загрузите код проекта:**
   ```python
   # Вариант 1: Клонировать из GitHub
   !git clone https://github.com/yourusername/pointnet2-segmentation.git
   %cd pointnet2-segmentation
   
   # Вариант 2: Загрузить файлы напрямую
   # Используйте вкладку Files в Colab для загрузки файлов проекта
   ```

3. **Установите зависимости:**
   ```python
   !pip install torch torchvision numpy scikit-learn tqdm matplotlib tensorboard -q
   ```

4. **Загрузите данные (опционально):**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Загрузите архив с PLY файлами
   ```

5. **Запустите обучение:**
   ```python
   !python train.py \
       --data_dir . \
       --area 3011 \
       --num_points 2048 \
       --batch_size 8 \
       --epochs 50 \
       --num_classes 13 \
       --device cuda
   ```

## Загрузка проекта на GitHub

### 1. Инициализация Git репозитория

```bash
cd /home/danil/Documents/GDEM
git init
git add .
git commit -m "Initial commit: PointNet++ для семантической сегментации"
```

### 2. Создание репозитория на GitHub

1. Зайдите на https://github.com
2. Создайте новый репозиторий (например, `pointnet2-segmentation`)
3. **НЕ** добавляйте README, .gitignore или лицензию (они уже есть)

### 3. Подключение и загрузка

```bash
git remote add origin https://github.com/yourusername/pointnet2-segmentation.git
git branch -M main
git push -u origin main
```

## Использование в Colab

### Вариант 1: Клонирование из GitHub

```python
!git clone https://github.com/yourusername/pointnet2-segmentation.git
%cd pointnet2-segmentation
!pip install -r requirements.txt
```

### Вариант 2: Прямая загрузка файлов

1. В Colab откройте вкладку "Files"
2. Загрузите все файлы проекта:
   - `dataset.py`
   - `pointnet2.py`
   - `train.py`
   - `test.py`
   - `metrics.py`
   - `visualize.py`
   - `requirements.txt`

### Загрузка данных в Colab

```python
from google.colab import files
import zipfile

# Загрузите архив с данными
uploaded = files.upload()

# Распакуйте
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
```

## Настройки для Colab

### Использование GPU

Colab предоставляет бесплатный GPU:
- Runtime → Change runtime type → GPU
- Проверка: `!nvidia-smi`

### Рекомендуемые параметры для Colab

```python
!python train.py \
    --data_dir . \
    --area 3011 \
    --num_points 2048 \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.001 \
    --num_classes 13 \
    --device cuda \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

### Сохранение результатов

```python
from google.colab import files
import zipfile
import os

# Создать архив с результатами
with zipfile.ZipFile('results.zip', 'w') as zipf:
    for root, dirs, files_list in os.walk('checkpoints'):
        for file in files_list:
            zipf.write(os.path.join(root, file))

# Скачать
files.download('results.zip')
```

## Преимущества Colab

✅ Бесплатный GPU (Tesla T4/K80)  
✅ Не нужно настраивать окружение  
✅ Легко делиться ноутбуками  
✅ Автоматическое сохранение в Google Drive  
✅ Доступ из любого места  

## Ограничения Colab

⚠️ Сессии ограничены по времени (12 часов)  
⚠️ GPU может быть недоступен в пиковые часы  
⚠️ Ограниченное дисковое пространство  
⚠️ Нужен интернет для работы  

## Сохранение в Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Сохранить результаты в Drive
!cp -r checkpoints /content/drive/MyDrive/pointnet2_results/
```

## Полный пример для Colab

```python
# 1. Установка
!pip install torch torchvision numpy scikit-learn tqdm matplotlib tensorboard -q

# 2. Клонирование репозитория
!git clone https://github.com/yourusername/pointnet2-segmentation.git
%cd pointnet2-segmentation

# 3. Загрузка данных (если нужно)
from google.colab import files
uploaded = files.upload()

# 4. Обучение
!python train.py --data_dir . --area 3011 --num_points 2048 --batch_size 8 --epochs 50 --device cuda

# 5. TensorBoard
%load_ext tensorboard
%tensorboard --logdir ./logs

# 6. Тестирование
!python test.py --checkpoint ./checkpoints/best_model.pth

# 7. Скачивание результатов
from google.colab import files
files.download('checkpoints/best_model.pth')
```

