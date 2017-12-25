import game
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
import os

population_size = 10		#must be an even number
fitness = [] 
current_models = []
generation = 1

def init_models():
	for i in range(population_size):
		rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
	
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
		
		
		current_models.append(model)
		fitness.append([0,0,0])	#W, L, D

	print ("Successfully Initialized Models!")

def load_models(gen):		#loads up saved models
	for i in range(population_size):
		current_models[i].load_weights("Model_Pool_Generation_" + str(gen) + "/model_new"+str(i)+".keras")
		
def save_models():
	if not os.path.exists("Model_Pool_Generation_" + str(generation)):
		os.makedirs("Model_Pool_Generation_" + str(generation))
	for xi in range(population_size):
		current_models[xi].save_weights("Model_Pool_Generation_" + str(generation) + "/model_new" + str(xi) + ".keras")
	print("Saved current pool!")

def normalizeFitness():
	#interpret fitness
	totalFitness = []
	totalGames = fitness[0][1] + fitness[0][2] + fitness[0][0]
	for ratio in fitness:
		score = 0
		score += (ratio[0]/totalGames)*5+(ratio[1]/totalGames)
		totalFitness.append(score)
		
	#normalize the data to 0-1
	minV = min(totalFitness)
	maxV = max(totalFitness)
	for i in range(population_size):
		totalFitness[i] = (totalFitness[i] - minV)/(maxV - minV)
	return
	
def breed(model1, model2):
	weights1 = current_models[model1].get_weights()
	weights2 = current_models[model2].get_weights()
def matchmake(model1, model2):
	return breed(model1, model2)
	
	
	
def evolve():
	total_fitness = normalizeFitness()
	
#fitness = [[1, 0, 17], [0, 1, 17], [0, 0, 18], [1, 0, 17], [1, 1, 16], [1, 1, 16], [0, 1, 17], [2, 0, 16], [1, 1, 16], [0, 2, 16]]
#evolve()
def trainGeneration():
	print("Beginning Training for Generation " + str(generation))
	print (str(population_size) + " models")
	
	for player1 in range(population_size):
		for player2 in range(population_size):
			if player1 == player2:
				print("Continue")
				continue
			players = (current_models[player1], current_models[player2])		#black, white
			
			game.reset()
			#gam = open("game.txt", "w")
			while(not game.board.is_game_over()):	#play the game
				bestMove = (None, -2)
				#Play the game
				generator = game.board.legal_moves
				for move in generator:
					game.push(move)			#evaluate the new position
					features = np.asarray(game.getFeatures())
					features = np.atleast_2d(features)
					score = players[int(game.turn())].predict(features, 1)
					#intv.write(str(score))	
					if score > bestMove[1]:
						bestMove = (move, score)	
					game.pop()	
				
				game.push(bestMove[0])
				#gam.write(str(bestMove[0]) + "\n")
				#gam.write(str(game.board.fen()) + "\n")
			#gam.close()
			result = game.board.result() 
			if result == '1-0':		#white (p2) won
				fitness[player2][0] += 1
				fitness[player1][1] += 1
			elif result == '0-1':
				fitness[player2][1] += 1
				fitness[player1][0] += 1
			else:
				fitness[player1][2] += 1
				fitness[player2][2] += 1
	print (fitness)
init_models()

model1 = current_models[0]
weights = model1.get_weights()
print(type(weights))
#print (weights)	
	
"""	
init_models()
save_models()
trainGeneration()
"""