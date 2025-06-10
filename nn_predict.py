import numpy as np
import json
import os  # Для os.path.join, если будет использоваться в __main__
import sys  # Для sys.path.append в __main__


# === Функции активации ===
def relu(x):
    """
    Реализует функцию активации Rectified Linear Unit (ReLU).
    f(x) = max(0, x)
    """
    return np.maximum(0, x)


def softmax(x):
    """
    Реализует функцию активации SoftMax.
    Преобразует вектор чисел в распределение вероятностей.
    Использует трюк для численной стабильности (вычитание максимума).
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# === Слой Flatten ===
def flatten(x):
    """
    Преобразует входной тензор в плоский (одномерный) вектор,
    сохраняя размерность батча.
    Ожидает вход (batch_size, height, width).
    """
    return x.reshape(x.shape[0], -1)


# === Полносвязный (Dense) слой ===
def dense(x, W, b):
    """
    Реализует полносвязный слой.
    Выполняет матричное умножение входных данных на веса и добавляет смещение.
    Ожидает вход (batch_size, input_features) и возвращает (batch_size, output_features).
    """
    return x @ W + b


# Основная функция для прямого прохода нейронной сети
def nn_forward_h5(model_arch_config_layers, weights, data):
    """
    Выполняет прямой проход нейронной сети на основе загруженной Keras-архитектуры
    (списка слоев) и весов, используя только NumPy.
    Поддерживает Flatten, Dense слои, и активации relu, softmax.

    Аргументы:
    model_arch_config_layers (list): Список словарей, представляющих слои модели (полученный
                                     из 'config']['layers'] полного .json файла Keras).
    weights (dict): Словарь весов модели (из .npz файла), где ключи - имена весов
                    (например, 'dense/kernel', 'dense/bias').
    data (numpy.ndarray): Входные данные для инференса. Ожидается форма (batch_size, 28, 28).

    Возвращает:
    numpy.ndarray: Выходные данные после прохода по сети.
    """
    x = data

    for layer_entry in model_arch_config_layers:
        ltype = layer_entry['class_name']
        cfg = layer_entry['config']

        lname = cfg['name']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            kernel_key = f"{lname}/kernel"
            bias_key = f"{lname}/bias"

            W = weights[kernel_key]
            b = weights[bias_key]

            x = dense(x, W, b)

            activation_fn_name = cfg.get("activation")
            if activation_fn_name == "relu":
                x = relu(x)
            elif activation_fn_name == "softmax":
                x = softmax(x)

    return x


# Главная функция для выполнения инференса (должна соответствовать nn_inference в model_test.py)
# Она принимает ПОЛНЫЙ JSON-объект архитектуры Keras
def nn_inference(full_model_arch_json, weights_dict, input_data):
    """
    Главная функция для выполнения инференса нейронной сети.
    Принимает полный JSON-объект архитектуры Keras и извлекает из него слои.

    Аргументы:
    full_model_arch_json (dict): Полный JSON-объект архитектуры Keras (результат json.load(f)).
    weights_dict (dict): Словарь весов модели (из .npz файла).
    input_data (numpy.ndarray): Входные данные для предсказания (форма: batch_size, 28, 28).

    Возвращает:
    numpy.ndarray: Выходные вероятности классов.
    """
    model_layers_config = full_model_arch_json['config']['layers']
    return nn_forward_h5(model_layers_config, weights_dict, input_data)


# Пример использования этого модуля (будет запускаться, если nn_predict.py запустить напрямую)
if __name__ == '__main__':
    sys.path.append('./utils')
    from mnist_reader import load_mnist

    MODEL_DIR = 'model'
    MODEL_ARCH_PATH = os.path.join(MODEL_DIR, 'fashion_mnist.json')
    MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'fashion_mnist.npz')
    DATA_PATH = './data/fashion'

    try:
        with open(MODEL_ARCH_PATH, 'r') as f:
            full_model_arch_json = json.load(f)

        weights = np.load(MODEL_WEIGHTS_PATH)

        x_test_flat, y_test = load_mnist(DATA_PATH, kind='t10k')

        x_test_normalized_reshaped = x_test_flat.astype('float32') / 255.0
        x_test_normalized_reshaped = x_test_normalized_reshaped.reshape((-1, 28, 28))

        print("\n--- Проверка инференса на NumPy ---")

        all_predictions = []
        for i in range(len(x_test_normalized_reshaped)):
            img_batch = np.expand_dims(x_test_normalized_reshaped[i], axis=0)
            output = nn_inference(full_model_arch_json, weights, img_batch)
            all_predictions.append(output)

        all_predictions_array = np.concatenate(all_predictions, axis=0)

        numpy_predicted_classes = np.argmax(all_predictions_array, axis=1)

        accuracy_numpy = np.mean(numpy_predicted_classes == y_test)
        print(f"Точность модели NumPy на тестовых данных: {accuracy_numpy:.4f}")

    except FileNotFoundError as e:
        print(
            f"Ошибка: Не найдены файлы модели или данных. Убедитесь, что 'main.py' был запущен и все пути правильные.")
        print(f"Отсутствующий файл: {e.filename}")
    except KeyError as e:
        print(f"Ошибка: Несоответствие в именах весов. Убедитесь, что 'main.py' сохраняет веса с правильными именами.")
        print(f"Отсутствующий ключ веса: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        import traceback

        traceback.print_exc()