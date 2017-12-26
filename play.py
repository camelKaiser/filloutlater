import tensorflow as tf
import game
import keras
import chess

playerColor = 'white'

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
model.add(Activation('sigmoid'))
model.compile(optimizer = rms, loss = 'mse')

def initModel(filename):
    model.load_weights(filename)
#initModel('Model_Pool_Generation_6/model_new7.keras'))

def get_next_move():
    generator = list(game.board.legal_moves)
    probabilities = []
    for move in generator:
        game.push(move)         #evaluate the new position
        features = np.asarray(game.getFeatures())
        features = np.atleast_2d(features)
        score = players[int(game.turn())].predict(features, 1)
                     
        probabilities.append(score[0][0]+1)     #add 1 to constrain hyperbolic tangent ouput to 0-2
        game.pop()  
        cumulative = sum(probabilities)
        for i in range(len(probabilities)):
            probabilities[i] /= cumulative
              
        return np.random.choice(generator, p=probabilities)
    

while not game.board.is_game_over():
    move = input("Your move? ")
    print ("pushing " + move)
    game.push_san(move)
    
    engineMove = get_next_move()
    print (engineMove.uci())
    game.push(engineMove)
    
    