from utils import *
import numpy as np
import time
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

# Define a DataValuation instance
dv = DataValuation(X_train_batch, y_train_batch, X_dev, y_dev)

# Run valuation
start_time = time.time()
vals_loo = dv.estimate(method="loo")
end_time = time.time()
time_loo = end_time - start_time
print("Time elapsed: {:.2f} seconds".format(time_loo))

start_time = time.time()
vals_tmc = dv.estimate(method="tmc-shapley")
end_time = time.time()
time_tmc = end_time - start_time
print("Time elapsed: {:.2f} seconds".format(time_tmc))

start_time = time.time()
vals_beta = dv.estimate(method="beta-shapley")
end_time = time.time()
time_beta = end_time - start_time
print("Time elapsed: {:.2f} seconds".format(time_beta))

start_time = time.time()
vals_cs = dv.estimate(method="cs-shapley")
end_time = time.time()
time_cs = end_time - start_time
print("Time elapsed: {:.2f} seconds".format(time_cs))

# Save the results
np.save('data/vals_time/vals_loo.npy', vals_loo)
np.save('data/vals_time/vals_tmc.npy', vals_tmc)
np.save('data/vals_time/vals_beta.npy', vals_beta)
np.save('data/vals_time/vals_cs.npy', vals_cs)
np.save('data/vals_time/time_loo.npy', time_loo)
np.save('data/vals_time/time_tmc.npy', time_tmc)
np.save('data/vals_time/time_beta.npy', time_beta)
np.save('data/vals_time/time_cs.npy', time_cs)