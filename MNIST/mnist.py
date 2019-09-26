import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

# load mnist dataset
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

#### Data Preprocessing
# Reshape image to vector
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

# One-hot encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
# Normalization
x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255

#### Model Construction
model = Sequential()
# Stack the layers into model
# 1st layer: Dense layer
model.add(Dense(500,input_shape=(784,)))
model.add(Activation('tanh'))
model.add(Dropout(0.8))  # 50% dropout
# 2nd layer: Dense layer
model.add(Dense(500))
model.add(Activation('tanh'))
model.add(Dropout(0.8)) # 50% dropout
# 3rd layer: Dense layer
model.add(Dense(500))
model.add(Activation('tanh'))
model.add(Dropout(0.8)) # 50% dropout
# Output layer: Dense layer with Dimension in 10 (one dimension for one category)
# Softmax: Widely used in classification problem, output the probability of each category
model.add(Dense(10))
model.add(Activation('softmax'))


# Define the loss function, optimizer (Method used in training, 'sgd' denotes stochastic gradient descent) and record the accuracy for each traning epoch
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Customize the model compile (For example, you want to customize the lr (aka learning rate))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

# Create a history instance of our little calss to record the callbacks information during training process
history = LossHistory()

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_Train_normalize, y_TrainOneHot, epochs=100, verbose=1, batch_size=200, shuffle=True, validation_data=(x_Test_normalize, y_TestOneHot), callbacks=[history])

# Use the small funny class written by us before to plot the learning curve
history.loss_plot('epoch')

scores = model.evaluate(x_Test_normalize,y_TestOneHot,batch_size=200,verbose=1)

# scores variable records two indicators of our model: scores[0] - loss; scores[1] - accuracy
print("The accuracy of the model in testing set is %f" % (scores[1]))