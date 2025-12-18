#!/bin/bash

# Скрипт для настройки правильного GitHub remote

echo "=== Настройка GitHub Remote ==="
echo ""
echo "Ваш коммит уже создан успешно!"
echo "Теперь нужно настроить правильный URL репозитория."
echo ""

# Удаляем старый remote если есть
git remote remove origin 2>/dev/null

echo "Создайте репозиторий на GitHub:"
echo "1. Зайдите на https://github.com"
echo "2. Нажмите '+' → 'New repository'"
echo "3. Название: pointnet2-segmentation (или любое другое)"
echo "4. НЕ добавляйте README, .gitignore или лицензию"
echo "5. Нажмите 'Create repository'"
echo ""

read -p "Введите ваш GitHub username: " username
read -p "Введите название репозитория (или нажмите Enter для 'pointnet2-segmentation'): " repo_name

if [ -z "$repo_name" ]; then
    repo_name="pointnet2-segmentation"
fi

repo_url="https://github.com/${username}/${repo_name}.git"

echo ""
echo "Добавляю remote: $repo_url"
git remote add origin "$repo_url"

echo ""
echo "Переименовываю ветку в main..."
git branch -M main

echo ""
echo "Загружаю на GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Успешно загружено на GitHub!"
    echo ""
    echo "Ваш репозиторий: https://github.com/${username}/${repo_name}"
    echo ""
    echo "Для использования в Colab:"
    echo "  !git clone $repo_url"
    echo "  %cd ${repo_name}"
else
    echo ""
    echo "Ошибка при загрузке. Проверьте:"
    echo "1. Репозиторий создан на GitHub"
    echo "2. Правильный username и название"
    echo "3. У вас есть права на запись"
fi

