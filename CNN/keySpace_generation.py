from keras.models import Model, load_model
from keras.layers import Dense
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import np_utils

# Load the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Build the model
model = Sequential()

# 1st convolution layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Additional layer for key generation
model.add(Dense(128, activation='tanh'))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Save the model
model.save("mnist_cnn_model.h5")


# Load the model
model = load_model("mnist_cnn_model.h5")

# Create a new model that shares all layers up to and including the new dense layer with the original model
key_gen_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def generate_key(image):
    # Make sure the image is normalized and in the right shape
    image = image / 255
    image = image.reshape((1, 28, 28, 1))

    # Generate the key
    key = key_gen_model.predict(image)

    # Scale the key's elements to be between 0 and 255
    key = (key - key.min()) / (key.max() - key.min()) * 255

    # Round the key's elements to the nearest integers
    key = np.round(key)

    # Reshape the key to a 1D array
    key = key.flatten()

    # Truncate the key to be the length of row*col of the image
    key = key[:28*28]

    # # Reshape the key back to a 2D array
    # key = key.reshape((28, 28))

    return key

# Test the function with a sample image
sample_image = X_train[0]
key = generate_key(sample_image)
