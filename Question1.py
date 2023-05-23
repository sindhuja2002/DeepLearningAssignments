import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define a function to count the number of model parameters
def count_parameters(model):
    return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

# Define the first CNN architecture
def cnn_architecture_1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# Define the second CNN architecture
def cnn_architecture_2():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# Define the third CNN architecture
def cnn_architecture_3():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# Train and evaluate the models
models = [cnn_architecture_1(), cnn_architecture_2(), cnn_architecture_3()]
accuracies = []

for i, model in enumerate(models):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    accuracies.append(accuracy)
    print(f"Accuracy for CNN architecture {i+1}: {accuracy}")

# Create a comparison table
comparison_table = {
    "CNN Architecture": ["CNN Architecture 1", "CNN Architecture 2", "CNN Architecture 3"],
    "Parameters": [count_parameters(models[0]), count_parameters(models[1]), count_parameters(models[2])],
    "Accuracy": accuracies
}

# Print the comparison table
print("\nComparison Table:")
print("{:<20} {:<15} {:<10}".format("CNN Architecture", "Parameters", "Accuracy"))
print("-" * 45)
for i in range(len(comparison_table["CNN Architecture"])):
    print("{:<20} {:<15} {:.2f}%".format(comparison_table["CNN Architecture"][i],
                                          comparison_table["Parameters"][i],
                                          comparison_table["Accuracy"][i] * 100))
