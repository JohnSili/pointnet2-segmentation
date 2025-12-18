#!/bin/bash

# Быстрая загрузка проекта на GitHub

echo "=== Загрузка проекта на GitHub ==="

# Проверка инициализации Git
if [ ! -d ".git" ]; then
    echo "Инициализация Git репозитория..."
    git init
fi

# Добавление всех файлов
echo "Добавление файлов..."
git add .

# Коммит
echo "Создание коммита..."
read -p "Введите сообщение коммита (или нажмите Enter для стандартного): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit: PointNet++ для семантической сегментации"
fi
git commit -m "$commit_msg"

# Проверка remote
if ! git remote | grep -q origin; then
    echo ""
    echo "Добавьте remote репозиторий:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/pointnet2-segmentation.git"
    echo ""
    read -p "Введите URL вашего GitHub репозитория (или нажмите Enter чтобы пропустить): " repo_url
    if [ ! -z "$repo_url" ]; then
        git remote add origin "$repo_url"
    fi
fi

# Загрузка на GitHub
if git remote | grep -q origin; then
    echo "Загрузка на GitHub..."
    git branch -M main
    git push -u origin main
    echo "✓ Проект загружен на GitHub!"
else
    echo ""
    echo "Remote не настроен. Выполните вручную:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/pointnet2-segmentation.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
fi

echo ""
echo "Для использования в Colab:"
echo "  !git clone https://github.com/YOUR_USERNAME/pointnet2-segmentation.git"
echo "  %cd pointnet2-segmentation"

