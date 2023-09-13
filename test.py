from utils import *
import numpy as np
import torch

train_size = 1000
train_start = 0
train_size_batch = 200
dev_size = 200
test_size = 300
np.random.seed(0)
torch.manual_seed(0)

X_train, y_train_clean, X_dev, y_dev, X_test, y_test = load_data(train_size, dev_size, test_size)

cnn_accuracy_pure, cnn_accuracies_pure = cnn_tests(X_train, y_train_clean, X_test, y_test, runs=16)
print('CNN accuracy on pure dataset: {}'.format(cnn_accuracy_pure))

logistic_accuracy_pure = logistic_test(X_train, y_train_clean, X_test, y_test)
print('Logistic Regression accuracy on pure dataset: {}'.format(logistic_accuracy_pure))
