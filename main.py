import tensorflow as tf
import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split

# Добавляем папку 'utils' в путь импорта, чтобы найти mnist_reader.py
current_script_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_script_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from mnist_reader import load_mnist

# --- Настройка путей ---
data_path = os.path.join(current_script_dir, 'data', 'fashion')
model_output_dir = os.path.join(current_script_dir, 'model')

# --- Загрузка и подготовка данных Fashion-MNIST ---
print(f"Загрузка ВСЕХ доступных данных (t10k) из '{data_path}'...")
try:
    x_full_flat, y_full = load_mnist(data_path, kind='t10k')
    print(f"Форма загруженных данных (t10k, плоская): {x_full_flat.shape}, {y_full.shape}")
except FileNotFoundError as e:
    print(
        f"Ошибка загрузки данных: Убедитесь, что файлы t10k-images-idx3-ubyte.gz и t10k-labels-idx1-ubyte.gz находятся в папке '{data_path}'.")
    print(f"Подробности ошибки: {e}")
    sys.exit(1)

x_full = x_full_flat.astype('float32') / 255.0
x_full = x_full.reshape((-1, 28, 28))
print(f"Форма загруженных данных (t10k, reshape для Keras): {x_full.shape}")

# --- Разделение данных ---
print("Разделение t10k данных на тренировочную (80%) и валидационную (20%) выборки...")
x_train, x_val, y_train, y_val = train_test_split(
    x_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)
print(f"Форма тренировочных данных: {x_train.shape}, {y_train.shape}")
print(f"Форма валидационных данных: {x_val.shape}, {y_val.shape}")

# --- Построение, компиляция и обучение модели TensorFlow ---
print("\nПостроение модели TensorFlow...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Начало обучения модели...")
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=1)
print("Обучение модели завершено.")

loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"Точность на валидационных данных (TensorFlow): {accuracy:.4f}")

# --- Сохранение модели и извлечение архитектуры/весов ---

if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

model_h5_path = os.path.join(model_output_dir, 'fashion_mnist.h5')
model.save(model_h5_path)
print(f"\nМодель сохранена в: {model_h5_path}")

loaded_model = tf.keras.models.load_model(model_h5_path)

model_json = loaded_model.to_json()
model_arch_path = os.path.join(model_output_dir, 'fashion_mnist.json')
with open(model_arch_path, 'w') as json_file:
    json_file.write(model_json)
print(f"Архитектура модели сохранена в {model_arch_path}")

# !!! НОВАЯ ИСПРАВЛЕННАЯ ЛОГИКА СОХРАНЕНИЯ ВЕСОВ В NPZ !!!
weights_to_save = {}
print("\n--- Извлечение и сохранение весов с ГАРАНТИРОВАННО УНИКАЛЬНЫМИ именами: ---")
for layer in loaded_model.layers:
    # Keras по умолчанию присваивает слоям имена типа 'dense', 'dense_1', 'flatten'
    # Мы используем эти имена для создания уникальных ключей.
    if isinstance(layer, tf.keras.layers.Dense):  # Проверяем, что это Dense слой
        layer_weights = layer.get_weights()

        # Веса (kernel)
        if len(layer_weights) > 0:
            key_name_kernel = f"{layer.name}/kernel"
            weights_to_save[key_name_kernel] = layer_weights[0]
            print(f"  Сохранен: '{key_name_kernel}' (Форма: {layer_weights[0].shape})")

        # Смещения (bias)
        if len(layer_weights) > 1:
            key_name_bias = f"{layer.name}/bias"
            weights_to_save[key_name_bias] = layer_weights[1]
            print(f"  Сохранен: '{key_name_bias}' (Форма: {layer_weights[1].shape})")
    elif isinstance(layer, tf.keras.layers.Flatten):
        # Flatten слой не имеет обучаемых весов, но для отладки можно вывести
        print(f"  Слой: '{layer.name}' (Тип: Flatten) - не имеет обучаемых весов.")
print("---------------------------------------------------------------------")

model_weights_npz_path = os.path.join(model_output_dir, 'fashion_mnist.npz')
np.savez(model_weights_npz_path, **weights_to_save)
print(f"Веса модели сохранены в {model_weights_npz_path}")

print("\nВсе необходимые файлы модели созданы в папке 'model'.")
print("Теперь вы можете запустить run_test.py для тестирования инференса NumPy.")