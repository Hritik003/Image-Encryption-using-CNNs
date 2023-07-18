# Image-Encryption-using-CNNs
An efficient Algorithm using 3D chaotic maps, Planet Domain, and key space generation using **CNN**.

# CNN - Convolutional Neural Network
It is widely used in Network architecture that learns  directly from the data. In this Project, we have designed an algorithm for Image Encryption that uses CNNs to generate efficient Keyspace for an Image. To train our model,
we have collected a dataset from the [Fashion-Mnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist). A convolutional 2-layered Network was trained to generate the Key Space.

# Approach
## Encryption Scheme
The algorithm begins with generating a chaotic sequence using a 3D tent map and, along with a logistic Map generating another 
<div align="center">
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/41d2c85b-3c5e-4095-8f00-267f8170851a)https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/41d2c85b-3c5e-4095-8f00-267f8170851a">     
</div>
sequence. Followed by some techniques like Row rotation, Column Rotation, and concatenating the decimal to binary converted pixel values from the image obtained from the Rotation techniques. 

## Planet Domain
Then the concatenated 1D array has been transformed
into the blocks for 3bit each, what we call a Planet, and then planets are xorred from the index Sorting method. This efficient method is what we call  "Virtual Planet Xorring." We have initialized each 3 Bit block as a planet and chose a 
rule from the table for the encoding scheme.

000 -> Mercury <br>
001 -> Venus <br>
010 -> Earth <br>
011 -> Mars<br>
100 -> Jupiter<br>
101 -> Saturn <br>
110 -> Uranus<br>
111 -> Neptune<br>

| ![space-1.jpg](https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/91a2a2ae-1c89-413c-9e36-c4a9a240f9b4) | 
|:--:| 
| _(Table : 12 rules out of 40320 rules for virtual planets and their respective 3-bit binary number which will be used for converting image from binary to VPD)_ |

## Key Space Generation using CNN
Our model has been trained with a two layered Convolutional Network, lets go step by step discussing it: <br>

1. **importing necessary modules:**
``` python
from keras.models import Model, load_model
from keras.layers import Dense
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import np_utils
```
Here, we import the required modules from Keras to build and train the model, load the Fashion MNIST dataset, and perform necessary data preprocessing.<br>
   
2. **Loading the Preprocessing the dataset:**
``` python
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
 
```
We load the Fashion MNIST dataset using the fashion_mnist.load_data() function. The dataset consists of 60,000 training images and 10,000 test images, 
each of size 28x28 pixels. We normalize the pixel values by dividing them by 255 to scale them to the range 0-1. 
The labels are one-hot encoded using np_utils.to_categorical(). We also reshape the input images to have a channel dimension (1 in this case) required for CNN input<br>

3. **Building the CNN model**
``` python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(10, activation='softmax'))

```
We create a sequential model object and add layers to it. The model starts with a convolutional layer (Conv2D) with 32 filters of size 3x3, followed by a ReLU activation function.
Then, a max pooling layer (MaxPooling2D) with a pool size of 2x2 is added. This pattern is repeated with 64 filters in the second convolutional layer.
After that, a Flatten layer is added to convert the 2D feature maps to a 1D feature vector. Next, two fully connected (Dense) layers are added with 128 units each,
one with a ReLU activation function and the other with a hyperbolic tangent (tanh) activation function. Finally,
an output layer with 10 units (corresponding to the 10 classes in Fashion MNIST) and a softmax activation function is added. <br>
   
4. **Compile, Train, and save the Model**
``` python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
model.save("mnist_cnn_model.h5")
```
We compile the model by specifying the loss function (categorical_crossentropy), optimizer (adam), and evaluation metric (accuracy) to be used during training.
We train the model using the fit function, passing the training data (X_train and y_train), validation data (X_test and y_test), number of epochs (10), and batch size (200).
The model will iterate over the training data for the specified number of epochs, updating the model's weights to minimize the defined loss function.
We train the model using the fit function, passing the training data (X_train and y_train), validation data (X_test and y_test), number of epochs (10), and batch size (200). 
The model will iterate over the training data for the specified number of epochs, updating the model's weights to minimize the defined loss function.

## Results
<div align="center">
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/096980e4-9b49-45b1-add8-5c477bb57958" width = 114px height = 114px>     
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/b00a35da-3bcb-49ec-ba3c-ed49c16ecbc1" width = 114px height = 114px>  
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/096980e4-9b49-45b1-add8-5c477bb57958" width = 114px height = 114px>  
</div>

<div align="center">
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/c2935a13-9053-4dfc-b728-6de504430915" width = 114px height = 114px>     
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/b00a35da-3bcb-49ec-ba3c-ed49c16ecbc1" width = 114px height = 114px>  
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/c2935a13-9053-4dfc-b728-6de504430915" width = 114px height = 114px>  
</div>

<div align="center">
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/8f03a68b-a72b-4dfb-99ef-d5c6a4a308cc" width = 114px height = 114px>     
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/b00a35da-3bcb-49ec-ba3c-ed49c16ecbc1" width = 114px height = 114px>  
    <img src="https://github.com/Hritik003/Image-Encryption-using-CNNs/assets/73677045/8f03a68b-a72b-4dfb-99ef-d5c6a4a308cc" width = 114px height = 114px>  
</div>








