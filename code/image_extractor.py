import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data = np.load("mars_for_students.npz")

training_set = data["training_set"]
X_train = training_set[:, 0].astype("uint8")
y_train = training_set[:, 1]

input_shape = X_train.shape[1:]
num_classes = len(np.unique(y_train))

# X
for i in range(len(X_train)):
	im = Image.fromarray(X_train[i], mode="L")
	im.save("images/X_train/"+str(i)+".jpg")
	
# y
def create_segmentation_colormap(num_classes):
    """
    Create a linear colormap using a predefined palette.
    Uses 'viridis' as default because it is perceptually uniform
    and works well for colorblindness.
    """
    return plt.cm.viridis(np.linspace(0, 1, num_classes))

def apply_colormap(label, colormap=None):
    """
    Apply the colormap to a label.
    """
    # Ensure label is 2D
    label = np.squeeze(label)

    # Apply the colormap
    colored = colormap[label.astype(int)]
    return colored

colormap = create_segmentation_colormap(num_classes)
mapped_labels = np.array([apply_colormap(label, colormap) for label in y_train])

for i in range(len(mapped_labels)):
	plt.imsave("images/y_train/"+str(i)+".jpg", mapped_labels[i])