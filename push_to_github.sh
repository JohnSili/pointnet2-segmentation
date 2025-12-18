#!/bin/bash

# Скрипт для загрузки кода на GitHub

echo "=== Загрузка на GitHub ==="
echo ""
echo "Репозиторий уже создан: https://github.com/JohnSili/pointnet2-segmentation"
echo ""

# Настройка remote
git remote remove origin 2>/dev/null
git remote add origin https://github.com/JohnSili/pointnet2-segmentation.git

echo "Попытка загрузки через GitHub CLI..."
gh repo sync JohnSili/pointnet2-segmentation --source . --force 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "Альтернативный способ:"
    echo "1. Зайдите на https://github.com/JohnSili/pointnet2-segmentation"
    echo "2. Скопируйте команды из раздела 'Quick setup'"
    echo ""
    echo "Или используйте:"
    echo "  git push -u origin main"
    echo ""
    echo "Если запросит пароль, используйте Personal Access Token:"
    echo "  https://github.com/settings/tokens"
fi

