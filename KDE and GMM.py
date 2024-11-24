import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

# Define data directory and categories
data_dir_training = "dataset/Training"
data_dir_testing = "dataset/Testing"
categories = ["glioma", "meningioma", "notumor", "pituitary"]
image_size = 256

# Load images function
def load_images(data_dir, categories, image_size):
    images = []
    labels = []
    for idx, category in enumerate(categories):
        path = os.path.join(data_dir, category)  # Fixed here
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (image_size, image_size)).flatten()
            images.append(img_resized)
            labels.append(idx)
    return np.array(images), np.array(labels)

# Load training and testing data
train_images, train_labels = load_images(data_dir_training, categories, image_size)
test_images, test_labels = load_images(data_dir_testing, categories, image_size)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_images)
test_pca = pca.transform(test_images)

# KDE Visualization
def plot_kde(data, labels, categories):
    plt.figure(figsize=(10, 8))
    for i, category in enumerate(categories):
        class_data = data[labels == i]
        kde = gaussian_kde(class_data.T)
        x, y = np.mgrid[class_data[:, 0].min():class_data[:, 0].max():100j,
                        class_data[:, 1].min():class_data[:, 1].max():100j]
        positions = np.vstack([x.ravel(), y.ravel()])
        density = kde(positions).reshape(x.shape)
        plt.contour(x, y, density, levels=5, alpha=0.7, label=f"{category} KDE")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Kernel Density Estimation (KDE)')
    plt.legend(categories)
    plt.grid()
    plt.show()

# GMM Visualization
def plot_gmm(data, labels, categories, n_components=4):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    preds = gmm.predict(data)

    plt.figure(figsize=(10, 8))
    for i, category in enumerate(categories):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], alpha=0.6, label=category)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=200, label='GMM Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Gaussian Mixture Model (GMM) Clustering')
    plt.legend()
    plt.grid()
    plt.show()

# Plot KDE for training data
plot_kde(train_pca, train_labels, categories)

# Plot GMM for training data
plot_gmm(train_pca, train_labels, categories)

