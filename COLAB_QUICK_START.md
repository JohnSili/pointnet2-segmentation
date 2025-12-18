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

## Данные уже включены в репозиторий! ✅

**Архив `data_for_colab.zip` (38MB, 500 PLY файлов) уже в репозитории.**

Ноутбук автоматически распакует данные при клонировании. Просто выполните ячейки по порядку!

### Если нужно загрузить свои данные:

```python
from google.colab import files
import zipfile

# Загрузите свой архив
uploaded = files.upload()

# Распакуйте
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Данные распакованы!")
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

