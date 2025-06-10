import os, json
import numpy as np
from nn_predict import nn_inference, softmax, relu
from utils import mnist_reader

YOUR_MODEL_PATH = 'model/fashion_mnist' # Default format is h5
#TF_MODEL_PATH = f'{YOUR_MODEL_PATH}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_PATH}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_PATH}.json'
OUTPUT_FILE = 'test_acc.txt'

def test_inference():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    acc = None
    try:
        x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

        # === Load weights and architecture ===
        weights = np.load(MODEL_WEIGHTS_PATH)
        with open(MODEL_ARCH_PATH) as f:
            model_arch = json.load(f)
            
        # Shuffle the test dataset
        indices = np.arange(x_test.shape[0])
        np.random.shuffle(indices)
        x_test_shuffled = x_test[indices]
        y_test_shuffled = y_test[indices]
        
        # Normalize input images
        normalized_X = x_test_shuffled / 255.0
        
        # Perform inference for all test images
        print('Classifying images...')
        outputs = np.array([
            nn_inference(model_arch, weights, np.expand_dims(img, axis=0))
            for img in normalized_X
        ])
        print('Done')

        # Get predictions using argmax
        predictions = np.argmax(outputs.squeeze(axis=1), axis=-1)

        # Calculate number of correct predictions
        correct = np.sum(predictions == y_test_shuffled)

        acc = correct / len(y_test_shuffled)
        print(f"Accuracy = {acc}")
        with open(OUTPUT_FILE, 'w') as file:
            file.write(str(acc))

    except Exception as e:
        print("Error! ", e)

    assert acc != None


def load_test_acc():
    acc = 0
    try:
        with open(OUTPUT_FILE, 'r') as file:
            # Read the first line and convert it to a float
            line = file.readline()
            acc = float(line.strip())
    except Exception as e:
        print("Error! ", e)
    
    return acc

def test_acc_80():
    acc = load_test_acc()
    assert acc >= 0.8

def test_acc_82():
    acc = load_test_acc()
    assert acc >= 0.82

def test_acc_84():
    acc = load_test_acc()
    assert acc >= 0.84

def test_acc_86():
    acc = load_test_acc()
    assert acc >= 0.86

def test_acc_88():
    acc = load_test_acc()
    assert acc >= 0.88

def test_acc_90():
    acc = load_test_acc()
    assert acc >= 0.9

def test_acc_91():
    acc = load_test_acc()
    assert acc >= 0.91

def test_acc_92():
    acc = load_test_acc()
    assert acc >= 0.92

def test_softmax():
    x = np.array([2.0, 1.0, 0.1])
    y = softmax(x)

    print("Softmax output:", y)
    print("Sum of outputs:", np.sum(y))  # Should be 1.0
    assert np.all(y >= 0) and np.all(y <= 1), "Output not in [0,1]"
    assert np.isclose(np.sum(y), 1.0), "Output does not sum to 1"

def test_relu():
    x = np.array([-2, -1, 0, 1, 2])
    y = sum(relu(x))

    assert y == 3
