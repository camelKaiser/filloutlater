import tensorflow as tf
import game
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(units = 296, input_dim = 148))
model.add(Activation('relu'))
model.add(Dense(units = 740))
model.add(Activation('relu'))
model.add(Dense(units = 740))
model.add(Activation('relu'))
model.add(Dense(units = 592))
model.add(Activation('relu'))
model.add(Dense(units = 296))
model.add(Activation('relu'))
model.add(Dense(units = 1))
model.add(Activation('tanh'))
model.compile(optimizer = rms, loss = 'mse')

def initModel(filename):
    model.load_weights(filename)
initModel('Model_Pool_Generation_6/model_new7.keras'))