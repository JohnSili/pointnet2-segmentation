# Быстрый старт в Google Colab

## Ваш репозиторий готов!

**URL:** https://github.com/JohnSili/pointnet2-segmentation

## Запуск в Colab (3 шага):

### 1. Откройте ноутбук в Colab

**Прямая ссылка:**
```
https://colab.research.google.com/github/JohnSili/pointnet2-segmentation/blob/main/colab_setup.ipynb
```

Или:
1. Откройте https://colab.research.google.com
2. File → Open Notebook
3. Вкладка "GitHub"
4. Введите: `JohnSili/pointnet2-segmentation`
5. Выберите `colab_setup.ipynb`

### 2. Включите GPU

Runtime → Change runtime type → Hardware accelerator → GPU

### 3. Запустите ячейки

Все готово! Просто выполняйте ячейки по порядку.

## Или клонируйте вручную:

```python
# Клонировать репозиторий
!git clone https://github.com/JohnSili/pointnet2-segmentation.git
%cd pointnet2-segmentation

# Установить зависимости
%pip install torch torchvision numpy scikit-learn tqdm matplotlib tensorboard -q

# Запустить обучение
!python train.py \
    --data_dir . \
    --area synthetic \
    --num_points 2048 \
    --batch_size 8 \
    --epochs 50 \
    --device cuda
```

## Загрузка ваших данных в Colab:

### Шаг 1: Создайте архив локально

```bash
cd /home/danil/Documents/GDEM
./prepare_data_for_colab.sh
# Или: zip -r data_for_colab.zip 3011-20251217T195928Z-1-001
```

### Шаг 2: Загрузите в Colab

В ноутбуке `colab_setup.ipynb` уже есть ячейка для загрузки. Или вручную:

```python
from google.colab import files
import zipfile

# Загрузите архив data_for_colab.zip
uploaded = files.upload()

# Распакуйте
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Данные распакованы!")
```

### Шаг 3: Запустите обучение

Ноутбук автоматически определит наличие данных и использует их:

```python
!python train.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --num_points 2048 \
    --batch_size 8 \
    --epochs 50 \
    --device cuda
```

**Подробная инструкция:** [DATA_UPLOAD.md](DATA_UPLOAD.md)

## Мониторинг обучения:

```python
%load_ext tensorboard
%tensorboard --logdir ./logs
```

## Скачивание результатов:

```python
from google.colab import files
files.download('checkpoints/best_model.pth')
```

---

**Готово! Проект полностью настроен для работы в Colab! 🚀**

