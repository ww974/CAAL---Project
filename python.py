from tensorflow.keras.datasets import mnist

# x_train: 60,000 images for training (shape: 60,000, 28, 28)
# y_train: their labels (digits from 0 to 9)
# x_test: 10,000 images for testing
# y_test: test labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Added extra dimension because our image is 2D (28,28), but the input in CNN is (height, width, channels).
# So the shape goes from (28,28) to (28,28,1)
x_train = x_train[..., None]
x_test = x_test[..., None]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

from tensorflow.keras import Input, Model, layers

# Input layer (shape: 28x28x1)
input_layer = Input(shape=(28, 28, 1))

# Convolutional layer (8 filters, each of size 5x5)
x = layers.Conv2D(filters=8, kernel_size=(5, 5), strides=1, padding='valid', activation='relu')(input_layer)

# Max Pooling layer (2x2 pooling with stride=2)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

# Flatten layer (flatten the 12x12x8 output into a vector of 1152 elements)
x = layers.Flatten()(x)  # Shape becomes (None, 1152)

# Fully connected (Dense) layer with 10 output units (for 10 classes)
output_layer = layers.Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Extract FC parameters (weights and biases from the last dense layer)
fc_layer = model.layers[-1]
W_fc, b_fc = fc_layer.get_weights()
print("W_fc shape:", W_fc.shape)  # Expected: (1152,10)
print("b_fc shape:", b_fc.shape)  # Expected: (10,)

# print(model.layers[0])  # Check the layer type
# print(model.layers[1])  # Check the layer type
# print(model.layers[2])  # Check the layer type

# #Extract weights and biases for the convolutional layer
# Step 1: Get the first Conv2D layer
conv_layer = model.layers[1]

weights, biases = conv_layer.get_weights()
print("conv_weight shape:", weights.shape)  # Expected: (8,1,5,5)
print("biases shape:", biases.shape)  # Expected: (8,)

# Saving Conv2D weights and biases to a text file
with open('conv_layer_weights.txt', 'w') as f:
    f.write("Conv2D Layer Weights (shape: {})\n".format(weights.shape))
    for i in range(weights.shape[-1]):  # Iterate over filters
        f.write(f"Filter {i + 1}:\n")
        f.write(str(weights[..., 0, i]) + "\n\n")  # Save each filter weights (5x5 matrix)

with open('conv_layer_biases.txt', 'w') as f:
    f.write("Conv2D Layer Biases (shape: {})\n".format(biases.shape))
    f.write(str(biases) + "\n")




# Saving Fully Connected (Dense) layer weights and biases to a text file
# with open('fc_layer_weights.txt', 'w') as f:
#     f.write("FC Layer Weights (shape: {})\n".format(W_fc.shape))
#     f.write(str(W_fc) + "\n")
import numpy as np
# np.savetxt('fc_layer_weights.txt', W_fc, fmt='%.6f')

with open('fc_layer_weights.txt', 'w') as f:
    f.write("FC Layer Weights (shape: {})\n\n".format(W_fc.shape))
    for i, row in enumerate(W_fc):
        # Format each weight in the row to 6 decimal places
        row_str = ' '.join(f"{w:.6f}" for w in row)
        f.write(f"Neuron {i} weights:\n{row_str}\n\n\n")  # 2 empty lines between rows


with open('fc_layer_biases.txt', 'w') as f:
    f.write("FC Layer Biases (shape: {})\n".format(b_fc.shape))
    f.write(str(b_fc) + "\n")

print("Weights and biases saved to text files.")


# Saving Conv2D layer weights in C++ array format
with open('conv_layer_weights_cpp.txt', 'w') as f:
    f.write("float conv_layer_weights[{}][{}][{}][{}] = {{\n".format(
        weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]))  # Shape: (5, 5, 1, 8)

    for i in range(weights.shape[0]):  # Iterate over filter height (5)
        for j in range(weights.shape[1]):  # Iterate over filter width (5)
            for k in range(weights.shape[2]):  # Iterate over input channels (1 in this case)
                f.write("    {")
                for l in range(weights.shape[3]):  # Iterate over filters (8)
                    if l == weights.shape[3] - 1:
                        f.write(f"{weights[i,j,k,l]:.6f}")
                    else:
                        f.write(f"{weights[i,j,k,l]:.6f}, ")
                if j == weights.shape[1] - 1 and k == weights.shape[2] - 1:
                    f.write(" }\n")
                else:
                    f.write(" },\n")
    f.write("};\n")


# Saving Conv2D layer biases in C++ array format
with open('conv_layer_biases_cpp.txt', 'w') as f:
    f.write("float conv_layer_biases[{}] = {{\n".format(biases.shape[0]))  # Shape: (8,)
    for i in range(biases.shape[0]):
        if i == biases.shape[0] - 1:
            f.write(f"    {biases[i]:.6f}\n")
        else:
            f.write(f"    {biases[i]:.6f},\n")
    f.write("};\n")

# Saving Fully Connected (Dense) layer weights in C++ array format
with open('fc_layer_weights_cpp.txt', 'w') as f:
    f.write("float fc_layer_weights[{}][{}] = {{\n".format(W_fc.shape[0], W_fc.shape[1]))  # Shape: (1152, 10)
    for i in range(W_fc.shape[0]):
        f.write("    {")
        for j in range(W_fc.shape[1]):
            if j == W_fc.shape[1] - 1:
                f.write(f" {W_fc[i,j]:.6f}")
            else:
                f.write(f" {W_fc[i,j]:.6f},")
        if i == W_fc.shape[0] - 1:
            f.write(" }\n")
        else:
            f.write(" },\n")
    f.write("};\n")

# Saving Fully Connected (Dense) layer biases in C++ array format
with open('fc_layer_biases_cpp.txt', 'w') as f:
    f.write("float fc_layer_biases[{}] = {{\n".format(b_fc.shape[0]))  # Shape: (10,)
    for i in range(b_fc.shape[0]):
        if i == b_fc.shape[0] - 1:
            f.write(f"    {b_fc[i]:.6f}\n")
        else:
            f.write(f"    {b_fc[i]:.6f},\n")
    f.write("};\n")

print("C++ array format weights and biases saved to text files.")

model.summary()

