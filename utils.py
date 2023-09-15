from matplotlib import pyplot as plt
from sklearn import linear_model
from keras.datasets import cifar10, mnist
import joblib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from valda.eval import data_removal
from valda.metrics import weighted_acc_drop
import seaborn as sns


def load_data(dataset_type, train_size, dev_size, test_size):
    # Check if the data directory exists
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    if dataset_type == 'MNIST':
        # Check if the MNIST dataset is already saved
        if os.path.exists('./dataset/mnist.pkl'):
            (X_train, y_train), (X_test, y_test) = joblib.load('./dataset/mnist.pkl')
        else:
            # Load MNIST data
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            # Save the MNIST dataset
            joblib.dump(((X_train, y_train), (X_test, y_test)), './dataset/mnist.pkl')

    elif dataset_type == 'CIFAR':
        # Check if the CIFAR dataset is already saved
        if os.path.exists('./dataset/cifar.pkl'):
            (X_train, y_train), (X_test, y_test) = joblib.load('./dataset/cifar.pkl')
        else:
            # Load CIFAR data
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()

            # Save the CIFAR dataset
            joblib.dump(((X_train, y_train), (X_test, y_test)), './dataset/cifar.pkl')

    else:
        raise ValueError("Invalid dataset type. Expected 'MNIST' or 'CIFAR'.")

    # Convert labels to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Flatten the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Split the data into train, dev, test sets
    X_dev = X_train[-dev_size:]
    y_dev = y_train[-dev_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def flip_labels(y, flip_fraction):
    num_classes = len(np.unique(y))

    # Determine the number of labels to flip
    num_flips = int(len(y) * flip_fraction)

    # Randomly choose indices of labels to flip
    flip_indices = np.random.choice(len(y), size=num_flips, replace=False)

    # Flip the chosen labels
    y_flipped = y.copy()
    for idx in flip_indices:
        y_flipped[idx] = (y[idx] + np.random.randint(1, num_classes)) % num_classes

    # Convert labels to integers
    y_flipped = y_flipped.astype(int)

    return y_flipped, flip_indices


class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit


def cnn_test(X_train, y_train, X_test, y_test):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reshape the data from 784 to 28x28
    if X_train.shape[1] == 784: # MNIST
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
        num_classes = 10
    elif X_train.shape[1] == 3072: # CIFAR
        X_train = X_train.reshape(-1, 3, 32, 32)
        X_test = X_test.reshape(-1, 3, 32, 32)
        num_classes = 10

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create the CNN model and move to GPU
    model = LeNet5(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    epochs = 64
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model
    outputs_test = model(X_test)
    _, predicted = torch.max(outputs_test, 1)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print('Accuracy of CNN model on {}: {:.2f}'.format(device, accuracy))

    return accuracy


def logistic_test(X_train, y_train, X_test, y_test):
    # Create the logistic regression model
    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)

    print('Accuracy of Logistic Regression model: {:.2f}'.format(accuracy))

    return accuracy


def cnn_tests(X_train, y_train, X_test, y_test, runs=8):
    cnn_accuracies = []

    for _ in range(runs):
        cnn_accuracy = cnn_test(X_train, y_train, X_test, y_test)
        cnn_accuracies.append(cnn_accuracy)

    cnn_accuracy = np.mean(cnn_accuracies)

    return cnn_accuracy, cnn_accuracies


def shap_comparison(X_train, y_train, X_test, y_test, remove_high_value=False, vals_rand=None, vals_loo=None, vals_tmc=None, vals_beta=None, vals_cs=None, vals_inf=None):
    if vals_rand is not None:
        accs_rand = data_removal(vals_rand, X_train, y_train, X_test, y_test, remove_high_value=remove_high_value)
    if vals_loo is not None:
        accs_loo = data_removal(vals_loo, X_train, y_train, X_test, y_test, remove_high_value=remove_high_value)
    if vals_tmc is not None:
        accs_tmc = data_removal(vals_tmc, X_train, y_train, X_test, y_test, remove_high_value=remove_high_value)
    if vals_beta is not None:
        accs_beta = data_removal(vals_beta, X_train, y_train, X_test, y_test, remove_high_value=remove_high_value)
    if vals_cs is not None:
        accs_cs = data_removal(vals_cs, X_train, y_train, X_test, y_test, remove_high_value=remove_high_value)
    if vals_inf is not None:
        accs_inf = data_removal(vals_inf, X_train, y_train, X_test, y_test, remove_high_value=remove_high_value)

    accs_len = len(X_train) + 1
    removal_percentages = [(i/accs_len) * 100 for i in range(accs_len)]

    if vals_rand is not None:
        plt.plot(removal_percentages, accs_rand, label='Rand')
    if vals_loo is not None:
        plt.plot(removal_percentages, accs_loo, label='LOO')
    if vals_tmc is not None:
        plt.plot(removal_percentages, accs_tmc, label='TMC')
    if vals_beta is not None:
        plt.plot(removal_percentages, accs_beta, label='Beta')
    if vals_cs is not None:
        plt.plot(removal_percentages, accs_cs, label='CS')
    if vals_inf is not None:
        plt.plot(removal_percentages, accs_inf, label='Inf')

    plt.xlabel('Data Removal Percentage (%)')
    plt.ylabel('Accuracy')
    if remove_high_value is False:
        plt.title('Accuracy vs. Ascending Data Removal Percentage')
    elif remove_high_value is True:
        plt.title('Accuracy vs. Descending Data Removal Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()

    res_rand, res_loo, res_tmc, res_beta, res_cs, res_inf = None, None, None, None, None, None
    if vals_rand is not None:
        res_rand = weighted_acc_drop(accs_rand)
    if vals_loo is not None:
        res_loo = weighted_acc_drop(accs_loo)
    if vals_tmc is not None:
        res_tmc = weighted_acc_drop(accs_tmc)
    if vals_beta is not None:
        res_beta = weighted_acc_drop(accs_beta)
    if vals_cs is not None:
        res_cs = weighted_acc_drop(accs_cs)
    if vals_inf is not None:
        res_inf = weighted_acc_drop(accs_inf)

    if vals_rand is not None:
        print("The weighted accuracy drop for Rand is {:.3f}".format(res_rand))
    if vals_loo is not None:
        print("The weighted accuracy drop for LOO is {:.3f}".format(res_loo))
    if vals_tmc is not None:
        print("The weighted accuracy drop for TMC is {:.3f}".format(res_tmc))
    if vals_beta is not None:
        print("The weighted accuracy drop for Beta is {:.3f}".format(res_beta))
    if vals_cs is not None:
        print("The weighted accuracy drop for CS is {:.3f}".format(res_cs))
    if vals_inf is not None:
        print("The weighted accuracy drop for Inf is {:.3f}".format(res_inf))

    return res_rand, res_loo, res_tmc, res_beta, res_cs, res_inf


def visualize_min_indices(X_raw, y_raw, min_indices):
    columns = 7

    num_images = len(min_indices)
    num_rows = num_images // columns + (num_images % columns > 0)

    fig, axs = plt.subplots(num_rows, columns, figsize=(15, 15))

    for i, min_index in enumerate(min_indices):
        min_image = X_raw[min_index]
        min_label = y_raw[min_index]

        row = i // columns
        col = i % columns
        axs[row, col].imshow(min_image.reshape(28, 28), cmap='gray')
        axs[row, col].set_title('Label: {}'.format(min_label))

    # Remove empty subplots
    if num_images % columns != 0:
        for col in range(num_images % columns, columns):
            fig.delaxes(axs[num_rows - 1, col])

    # matplotlib.use('TkAgg')
    plt.tight_layout()
    plt.show()


def create_confusion_matrix(total_min_indices, flip_indices, len_X):
    matrix = np.zeros((2, 2), dtype=int)

    for i in range(len_X):
        if i in flip_indices and i in total_min_indices:
            matrix[0][0] += 1  # TP
        elif i not in flip_indices and i in total_min_indices:
            matrix[0][1] += 1  # FP
        elif i in flip_indices and i not in total_min_indices:
            matrix[1][0] += 1  # FN
        else:
            matrix[1][1] += 1  # TN

    return matrix


def plot_confusion_matrix_with_metrics(matrix):
    # Calculate metrics
    accuracy, precision, recall, f1_score, specificity = calculate_metrics(matrix)

    fig = plt.figure(figsize=(3, 4))

    # Create the heatmap axis occupying the top part of the figure
    ax1 = fig.add_axes([0, 0.25, 1, 0.75])
    sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', ax=ax1)
    ax1.set_aspect(1)
    ax1.set_xticks(np.arange(2))
    ax1.set_yticks(np.arange(2))
    ax1.set_xticklabels(['Flip', 'Not Flip'])
    ax1.set_yticklabels(['Neg', 'Not Neg'])
    ax1.set_title("Confusion Matrix")

    # Create an axis for the metrics text directly below the heatmap
    ax2 = fig.add_axes([0, 0, 1, 0.25])

    # Display the metrics on the bottom subplot
    metrics_text = f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1_score:.2f}\nSpecificity: {specificity:.2f}'
    ax2.axis('off')
    ax2.text(0.4, 0.5, metrics_text, ha="center", va="center", transform=ax2.transAxes)

    plt.show()


def calculate_metrics(matrix):
    TP = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TN = matrix[1][1]

    # Accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    # Precision
    precision = TP / (TP + FP)

    # Recall (Sensitivity)
    recall = TP / (TP + FN)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Specificity
    specificity = TN / (TN + FP)

    return accuracy, precision, recall, f1_score, specificity


def delete_data(X, y, min_indices, delete_ratio):
    num_delete = int(len(min_indices) * delete_ratio)

    delete_indices = min_indices[:num_delete]

    X_cleaned = np.delete(X, delete_indices, axis=0)
    y_cleaned = np.delete(y, delete_indices, axis=0)

    return X_cleaned, y_cleaned


def visualize_accuracy(delete_ratios, cnn_accs, logistic_accs, cnn_accs_pure, logistic_acc_pure):
    cnn_means = [np.mean(acc) for acc in cnn_accs]
    cnn_stds = [np.std(acc) for acc in cnn_accs]

    cnn_pure_mean = np.mean(cnn_accs_pure)
    cnn_pure_std = np.std(cnn_accs_pure)

    plt.errorbar(delete_ratios, cnn_means, yerr=cnn_stds, label='CNN (Poisoned Data)')
    plt.plot(delete_ratios, logistic_accs, label='Logistic Regression (Poisoned Data)')
    plt.axhline(y=logistic_acc_pure, color='m', linestyle='--', label='Logistic Regression (Pure Data)')
    plt.axhspan(cnn_pure_mean - cnn_pure_std, cnn_pure_mean + cnn_pure_std, facecolor='c', alpha=0.2)
    plt.axhline(y=cnn_pure_mean, color='c', linestyle='--', label='CNN (Pure Data)')

    plt.xlabel('Delete Ratio')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Delete Ratio')
    plt.legend()
    plt.grid()

    # Customize the ticks on the x and y axis
    plt.yticks(np.arange(0.4, 1.05, 0.05))

    plt.show()
