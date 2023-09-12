from utils import *
import numpy as np
import time
import torch
from valda.valuation import DataValuation

train_total = 1000
train_start = 0
train_size = 200
dev_size = 100
test_size = 300
np.random.seed(0)
torch.manual_seed(0)

# Load data
X_train, y_train_clean, X_dev, y_dev, X_test, y_test = load_data(train_total, dev_size, test_size)
y_train, flip_indices = flip_labels(y_train_clean, flip_fraction=0.3)

# Create a batch of training data
X_train_batch = X_train[train_start:train_start + train_size]
y_train_batch = y_train[train_start:train_start + train_size]

# Define a DataValuation instance
dv = DataValuation(X_train_batch, y_train_batch, X_dev, y_dev)

# Run with the default setting:
# 1. Logistic regression as the classifier
# 2. Prediction accuracy as the value measurement

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

start_time = time.time()
vals_inf = dv.estimate(method="inf_function")
end_time = time.time()
time_inf = end_time - start_time
print("Time elapsed: {:.2f} seconds".format(time_inf))

# Create vals_rand dictionary
vals_rand = {}
for i in range(len(X_train_batch)):
    vals_rand[i] = np.random.rand()

# Plot the results
shap_comparison(X_train, y_train, X_test, y_test,
                vals_loo=vals_loo,
                vals_tmc=vals_tmc,
                vals_beta=vals_beta,
                vals_cs=vals_cs,
                vals_inf=vals_inf,
                vals_rand=vals_rand)

