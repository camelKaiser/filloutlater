#Winston Li 12-25-17
#Trains our ENNs
import csv
import game
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
import os

population_size = 50    
fitness = [] 
current_models = []
generation = 1

def init_models():      #initialize initial poppulation
    for i in range(population_size):
        rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  #this part doesn't matter
    
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
        fitness.append([0,0,0]) #Win, Loss, Draw

    print ("Successfully Initialized Models!")

def load_models(gen):       #loads up saved models
    #TODO add custom model locations
    for i in range(population_size):
        current_models[i].load_weights("Model_Pool_Generation_" + str(gen) + "/model_new"+str(i)+".keras")
        
def save_models():      #Saves each generation to it's own folder. If storage is an issue remove generation from the save path
    if not os.path.exists("Model_Pool_Generation_" + str(generation)):
        os.makedirs("Model_Pool_Generation_" + str(generation))
    for xi in range(population_size):
        current_models[xi].save_weights("Model_Pool_Generation_" + str(generation) + "/model_new" + str(xi) + ".keras")
    print("Saved current pool!")

def save_fitness():     #saves fitness values 
    with open("Model_Pool_Generation_" + str(generation) + '/fitness.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(fitness)):
            writer.writerow([i, fitness[i][0], fitness[i][1], fitness[i][2]])
        
def normalizeFitness():         #modular function that can be tweaked as needed
    #interpret fitness (win loss draw ratios) as decided
    totalFitness = []
    totalGames = float(fitness[0][1]) + fitness[0][2] + fitness[0][0]
    for ratio in fitness:
        score = 0
        score += (ratio[0]/totalGames)*5+(ratio[1]/totalGames)
        totalFitness.append(score)
    #normalize the data to 0-1
    minV = min(totalFitness)
    maxV = max(totalFitness)
    for i in range(population_size):
        totalFitness[i] = (totalFitness[i] - minV)/(maxV - minV)
    return totalFitness

def mutate(weights): #randomly mutate a set of weights
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if random.uniform(0, 1) > 0.85:
                change = random.uniform(-0.5,0.5)
                weights[xi][yi] += change
    return weights
    
def breed(model1, model2):      #breed two arbitrary models randomly together
    weights1 = current_models[model1].get_weights()
    weights2 = current_models[model2].get_weights()
    child = []
    for layer in range(len(weights1)):
        if random.randint(0,1):
            child.append(weights1[layer])
        else:
            child.append(weights2[layer])
    return child
        
def matchmake(model1, model2):      #sounds cooler than "breed"
    return breed(model1, model2)
    
    
def evolve():
    print("Beginning Evolution")
    newPopulation = []
    total_fitness = normalizeFitness()
    for select in range(population_size):
        parent1 = random.uniform(0, 1)
        parent2 = random.uniform(0, 1)
        idx1 = -1
        idx2 = -1
        print(total_fitness)
        for idxx in range(population_size):     #better models have > chance for kids
            if total_fitness[idxx] >= parent1:
                idx1 = idxx
                break
        for idxx in range(population_size):
            if total_fitness[idxx] >= parent2:
                idx2 = idxx
                break
        new_weights = matchmake(idx1, idx2)     #spawn and corrupt the child
        new_weights = mutate(new_weights)
        newPopulation.append(new_weights)
        
    #cull the old
    for i in range(population_size):
        current_models[i].set_weights(newPopulation[i])
   
    print("Evolution finished for gen " + str(generation))
    
def trainGeneration():
    print("Beginning Training for Generation " + str(generation))
    print (str(population_size) + " models")
    
    for player1 in range(population_size):          #Simulate a round robin where every possible model/color match occurs
        for player2 in range(population_size):
            if player1 == player2:
                continue
            players = (current_models[player1], current_models[player2])        #black, white
            
            game.reset()
            while(not game.board.is_game_over()):   #play the game
                bestMove = (None, -2)
                #Play the game
                generator = game.board.legal_moves
                for move in generator:
                    game.push(move)         #evaluate the new position
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
            if result == '1-0':     #white (p2) won
                fitness[player2][0] += 1
                fitness[player1][1] += 1
                print ("Model " + str(player1) + " (black) just lost to model " + str(player2) + " (white)")
            elif result == '0-1':
                fitness[player2][1] += 1
                fitness[player1][0] += 1
                print ("Model " + str(player1) + " (black) just beat model " + str(player2) + " (white)")
            else:
                fitness[player1][2] += 1
                fitness[player2][2] += 1
                print ("Model " + str(player1) + " (black) just drew model " + str(player2) + " (white)")
    print ("Finished Generation " + str(generation))
    save_fitness()

init_models()
save_models()
while (generation > 0):
    trainGeneration()
    evolve()
    generation += 1
    save_models()