import os
import sys

# Добавляем корневую папку проекта в sys.path, чтобы импортировать model_test
sys.path.append('.')
import model_test

print("Запуск test_inference из run_test.py...")
model_test.test_inference()
print("Тест завершен.")