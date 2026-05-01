import numpy as np


def load_data():
    X = np.zeros((100, 4))
    y = np.zeros((100,))
    print('load_data | X.shape=(100, 4) | y.shape=(100,)')
    return {'X': X, 'y': y}


def get_batch(data, batch_size=32):
    idx = np.random.randint(0, len(data['X']) - batch_size)
    X_batch = data['X'][idx : idx + batch_size]
    y_batch = data['y'][idx : idx + batch_size]
    assert X_batch.shape == (batch_size, 4)
    assert y_batch.shape == (batch_size,)
    print('get_batch | X_batch.shape=(32, 4) | y_batch.shape=(32,)')
    return {'X': X_batch, 'y': y_batch}
