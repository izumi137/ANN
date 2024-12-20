import numpy as np
import requests
import os

# URLs for Fashion MNIST data stored on GitHub (update links as needed)
github_base_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Download and save the dataset
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {dest_path}")
    else:
        print(f"Failed to download: {url}")

# Ensure output directory exists
output_dir = "./fashion_mnist/"
os.makedirs(output_dir, exist_ok=True)

# Download all files
for name, file in files.items():
    file_url = github_base_url + file
    dest_path = os.path.join(output_dir, file)
    download_file(file_url, dest_path)

# Helper to load Fashion MNIST dataset from .gz files
import gzip

def load_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

def load_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

# Load the data
train_images = load_images(os.path.join(output_dir, files["train_images"]))
train_labels = load_labels(os.path.join(output_dir, files["train_labels"]))
test_images = load_images(os.path.join(output_dir, files["test_images"]))
test_labels = load_labels(os.path.join(output_dir, files["test_labels"]))

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Example: Save the first training image to a .txt file
def save_image_to_txt(image, label, file_path):
    with open(file_path, 'w') as f:
        for row in image:
            f.write(" ".join(map(str, row)))
            f.write(" ")
        f.write(str(label))

save_image_to_txt(train_images[0], train_labels[0], "./fashion_mnist_sample_image.txt")
print("Saved sample image to fashion_mnist_sample_image.txt")



# Save all images to a single .txt file

output_file = "./train.txt"

with open(output_file, 'w') as f:
    for i, image in enumerate(train_images):
        for row in image:
            f.write(" ".join(map(str, row)))
            f.write(" ")
        f.write(str(train_labels[i]))
        # Add newline between images
        f.write("\n")
    print(f"Saved all images to {output_file}")


output_file = "./test.txt"

with open(output_file, 'w') as f:
    for i, image in enumerate(test_images):
        for row in image:
            f.write(" ".join(map(str, row)))
            f.write(" ")
        f.write(str(test_labels[i]))
        # Add newline between images
        f.write("\n")
    print(f"Saved all images to {output_file}")