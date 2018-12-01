import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt


def plot_history(history):
    """Plot Results of Keras training"""
    plt.style.use('fivethirtyeight')
    epochs = list(range(1, len(history['loss']) + 1))
    plt.figure(figsize = (18, 6))
    
    # Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], '-o', ms = 10, label = "Training Loss")
    plt.plot(epochs, history['val_loss'], '-*',  ms = 10, label = "Validation Loss")
    plt.legend(); 
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Losses');
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['acc'], '-o', ms = 10, label = 'Training Acc')
    plt.plot(epochs, history['val_acc'], '-*',  ms = 10, label = "Validation Acc")
    plt.legend()
    plt.xlabel('Epoch'); plt.ylabel('Acc')
    plt.title('Accuracy');
    
    plt.suptitle('Training Curves', y= 1.05)


def get_options(slack):
    command_dict = {'functions': {},
                    'attributes': {}}

    # Modules
    for d in dir(slack):
        if not d.startswith('_'):
            command_dict['functions'][d] = []
            command_dict['attributes'][d] = []
            # Iterate through methods and attributes
            for dd in dir(getattr(slack, d)):
                if not dd.startswith('_'):
                    # List of methods and attributes
                    l = dir(getattr(getattr(slack, d), dd))
                    # Method (function)
                    if '__call__' in l:
                        command_dict['functions'][d].append(dd)
                    # Attributes
                    else:
                        command_dict['attributes'][d].append(dd)
                        
    return command_dict

def get_data_and_model():
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    return x_train, x_test, y_train, y_test, model