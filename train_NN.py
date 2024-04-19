from Layers import Layer_Dense
from Activation_Funcs import Activation_ReLU
from Activation_Funcs import Activation_Softmax
from Losses import Loss_CategoricalCrossentropy
from Optimizers import Optimizer_Adam
from Accuracies import Accuracy_Categorical
from Model import Model
from dataset_preparation import create_data_mnist
from dataset_preparation import load_mnist_dataset
import numpy as np

'''
import nnfs
nnfs.init()
# The nnfs.init() does three things: it sets the random seed to 0 (by the default), creates a float32 dtype default, and overrides the original dot product from NumPy. 
# All of these are meant to ensure repeatable results for following along.
'''


# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]
# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
127.5) / 127.5

# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
loss = Loss_CategoricalCrossentropy(),
optimizer = Optimizer_Adam(decay=1e-3),
accuracy = Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train and evaluate the model
model.train(X, y, validation_data=(X_test, y_test),
epochs=10, batch_size=128, print_every=100)


# save the model to local path
model.save('fashion_mnist2.model')