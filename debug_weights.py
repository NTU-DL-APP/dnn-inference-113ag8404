import tensorflow as tf
import numpy as np
import os
import json

# --- Конфигурация путей ---
# Этот скрипт должен находиться в корневой папке вашего проекта.
# Он будет искать модель .h5 в папке 'model'.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_script_dir, 'model')
model_h5_path = os.path.join(model_dir, 'fashion_mnist.h5')

# Мы сохраним отладочные веса в новый файл, чтобы избежать путаницы.
debug_weights_npz_path = os.path.join(model_dir, 'debug_weights_output.npz')

print(f"--- Запуск debug_weights.py ---")
print(f"Попытка загрузить модель Keras из: {model_h5_path}")

try:
    # 1. Загружаем обученную Keras-модель из файла .h5
    # ЭТО ТА ЖЕ МОДЕЛЬ, КОТОРУЮ СОЗДАЕТ main.py
    loaded_model = tf.keras.models.load_model(model_h5_path)
    print("Модель Keras успешно загружена.")

    # 2. Извлекаем и выводим имена весов, используя ТОЧНО ТУ ЖЕ ЛОГИКУ, что и в main.py
    weights_to_save = {}
    print("\n--- Извлеченные имена весов (из loaded_model.weights): ---")
    for weight_tensor in loaded_model.weights:
        # ЭТА СТРОКА ГЕНЕРИРУЕТ ИМЯ КЛЮЧА
        clean_weight_name = weight_tensor.name.split(':')[0]
        weights_to_save[clean_weight_name] = weight_tensor.numpy()
        print(f"  - Оригинальное имя веса: '{weight_tensor.name}'  ->  Сгенерированный ключ NPZ: '{clean_weight_name}' (Форма: {weight_tensor.numpy().shape})")
    print("---------------------------------------------------------")

    # 3. Сохраняем эти извлеченные веса в НОВЫЙ .npz файл
    np.savez(debug_weights_npz_path, **weights_to_save)
    print(f"\nВеса сохранены во временный файл: {debug_weights_npz_path}")

    # 4. Сразу же загружаем этот временный .npz файл и проверяем его ключи
    print(f"\n--- Проверка ключей в '{debug_weights_npz_path}' ---")
    loaded_debug_weights = np.load(debug_weights_npz_path)
    print("Ключи, найденные в debug_weights_output.npz:", loaded_debug_weights.files)

    # Проверяем наличие ожидаемых ключей
    if 'dense/kernel' in loaded_debug_weights.files and 'dense/bias' in loaded_debug_weights.files:
        print("✅ Успех: 'dense/kernel' и 'dense/bias' найдены в debug_weights_output.npz!")
    else:
        print("❌ Ошибка: 'dense/kernel' или 'dense/bias' НЕ найдены в debug_weights_output.npz!")
        print("Это указывает на более глубокую проблему с тем, как TensorFlow присваивает имена весам или как np.savez работает в вашей системе.")

    loaded_debug_weights.close()

except FileNotFoundError:
    print(f"Ошибка: Файл модели Keras '{model_h5_path}' не найден.")
    print("Пожалуйста, убедитесь, что 'main.py' был успешно запущен и создал файл fashion_mnist.h5 в папке 'model'.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Завершение debug_weights.py ---")