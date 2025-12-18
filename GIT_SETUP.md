# Инструкция по загрузке на GitHub

## Шаг 1: Подготовка репозитория

```bash
cd /home/danil/Documents/GDEM

# Инициализация Git (если еще не сделано)
git init

# Добавление всех файлов
git add .

# Первый коммит
git commit -m "Initial commit: PointNet++ для семантической сегментации облака точек"
```

## Шаг 2: Создание репозитория на GitHub

1. Зайдите на https://github.com
2. Нажмите "+" → "New repository"
3. Название: `pointnet2-segmentation` (или любое другое)
4. Описание: "PointNet++ для семантической сегментации облака точек"
5. Выберите Public или Private
6. **НЕ** добавляйте README, .gitignore или лицензию (они уже есть в проекте)
7. Нажмите "Create repository"

## Шаг 3: Подключение к GitHub

```bash
# Добавьте remote (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pointnet2-segmentation.git

# Переименуйте ветку в main (если нужно)
git branch -M main

# Загрузите код
git push -u origin main
```

## Шаг 4: Проверка

Откройте ваш репозиторий на GitHub - все файлы должны быть там.

## Использование в Colab

После загрузки на GitHub, в Colab:

```python
# Клонировать репозиторий
!git clone https://github.com/YOUR_USERNAME/pointnet2-segmentation.git
%cd pointnet2-segmentation

# Установить зависимости
!pip install -r requirements.txt

# Запустить обучение
!python train.py --data_dir . --area 3011 --num_points 2048 --batch_size 8 --epochs 50 --device cuda
```

## Обновление репозитория

Когда вносите изменения:

```bash
git add .
git commit -m "Описание изменений"
git push
```

## Что НЕ загружается на GitHub

Благодаря `.gitignore`, следующие файлы/папки НЕ будут загружены:
- `venv/` - виртуальное окружение
- `checkpoints/` - сохраненные модели
- `logs/` - логи TensorBoard
- `visualizations/` - результаты визуализации
- `__pycache__/` - кэш Python
- Данные (`.ply`, `.txt`, `.npy` файлы)

## Рекомендации

1. **Добавьте описание проекта** в README.md
2. **Добавьте лицензию** (если нужно)
3. **Создайте теги** для версий:
   ```bash
   git tag -a v1.0 -m "Первая версия"
   git push origin v1.0
   ```

## Прямая ссылка для Colab

После загрузки на GitHub, вы можете использовать прямую ссылку:

```
https://colab.research.google.com/github/YOUR_USERNAME/pointnet2-segmentation/blob/main/colab_setup.ipynb
```

Или создайте кнопку в README.md:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/pointnet2-segmentation/blob/main/colab_setup.ipynb)
```

