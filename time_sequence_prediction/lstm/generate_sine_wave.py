import numpy as np
import torch
import os

folder = os.environ['DATA_DIRECTORY']
data_folder = folder + '/time_series_prediction'
os.makedirs(data_folder, exist_ok=True)

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')

torch.save(data, open(data_folder + '/traindata.pt', 'wb'))
