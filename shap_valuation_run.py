from utils import *
import numpy as np
import torch
from valda.valuation import DataValuation

train_size = 1000
train_start = 0
train_size_batch = 200
dev_size = 300
test_size = 1000
np.random.seed(0)
torch.manual_seed(0)

# Load data
X_train, y_train_clean, X_dev, y_dev, X_test, y_test = load_data(train_size, dev_size, test_size)
y_train, flip_indices = flip_labels(y_train_clean, flip_fraction=0.3)

X_train_batch = X_train[train_start:train_start+train_size_batch]
y_train_batch = y_train[train_start:train_start+train_size_batch]

# Run valuation
dv = DataValuation(X_train_batch, y_train_batch, X_dev, y_dev)
vals_tmc = dv.estimate(method="tmc-shapley")

# Save the indices of the data points with negative values
min_indices_batch = [key for key, value in vals_tmc.items() if value <= 0]
min_indices_batch = np.array(min_indices_batch) + train_start
file_path = f'./data/min_indices_{str(train_start).zfill(3)}-{str(train_start+train_size_batch-1).zfill(3)}_{dev_size}.npy'
np.save(file_path, min_indices_batch)