from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt # grafics and images.

# fetch_openml: download MNIST: 70,000 images of handwritten digits (28x28 pixels).
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# X holds the images, y holds the correct digit for each one.
X, y = mnist.data, mnist.target

# each image is a row of 784 numbers (28x28 pixels, values 0 to 255).
print("Images shape:", X.shape)
print("Labels shape:", y.shape)

# show the first 10 images with their labels
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(28, 28), cmap="gray") # reshapes the row of 784 pixels back into a 28x28 square.
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show() # without it, there's no 10-digit tab.