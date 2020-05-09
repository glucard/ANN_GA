import numpy as np
import random as rd
from operator import itemgetter
import time

class Roleta():
    def rolar(objects, ratings = None):
        if ratings == None:
            return rd.choice(objects)
        max_n_choices = sum(ratings)
        n_choice = rd.randrange(1,max_n_choices+1)
        i = 0
        while n_choice > 0:
            n_choice -= ratings[i]
            choice = objects[i]
            i+=1
        return choice
    
    def torneio(rating_objects, ratings = None, limiter = None):
        rating_objects.sort(key=lambda x: x[0])
        ratings  = []
        objects = []
        if limiter == None:
            limiter_thing = 0
        else:
            limiter_thing = limiter
        for i in range(len(rating_objects[-limiter_thing:])):
            ratings.append(i+1)
            objects.append(rating_objects[-limiter_thing+i][1])
        return Roleta.rolar(objects, ratings)
    
    def pior(rating_objects, ratings = None):
        rating_objects.sort(key=lambda x: x[0])
        return rating_objects[-1][1]
    
    def melhor(rating_objects, ratings = None):
        rating_objects.sort(key=lambda x: x[0])
        return rating_objects[0][1]
    
    def segundo(rating_objects, ratings = None):
        rating_objects.sort(key=lambda x: x[0])
        return rating_objects[1][1]

def ReLU(x, derivative = False):
    if(derivative):
        return 1. * (x > 0)
    return x * (x > 0)

def sigmoid (x, derivative = False):
    if(derivative):
        return x * (1 - x)
    return 1/(1 + np.exp(-x))

def crossoverMatrix(a, b):
    an = a.copy()
    bn = b.copy()
    tempa = an.copy()
    tempb = bn.copy()
    try:
        cutx = rd.randrange(0,len(an)-1)
    except:
        cutx = 0
    cuty = rd.randrange(len(an[0]))
    an[:cutx], bn[:cutx] = tempb[:cutx].copy(), tempa[:cutx].copy()
    an[cutx][:cuty], bn[cutx][:cuty] = tempb[cutx][:cuty].copy(), tempa[cutx][:cuty].copy()
    return [an, bn]

def mutateMatrix(m, rate):
    n = m.copy()
    mutation = 0.1
    if rd.randrange(0,100) < rate:
        try:
            x = rd.randrange(0,len(n)-1)
        except:
            x = 0
        y = rd.randrange(0,len(n[0]))
        roll = rd.randrange(0,4)
        if roll == 0:
            z = 1-mutation
        elif roll == 1:
            z = 1+mutation
        elif roll == 2:
            z = -1-mutation
        else:
            z = -1+mutation
        n[x][y] = n[x][y] * z
        if n[x][y] > 2:
            n[x][y] = 2
        elif n[x][y] < -2:
            n[x][y] = -2
        elif 0 > n[x][y] > -0.001:
            n[x][y] = -0.001
        elif 0 < n[x][y] < 0.001:
            n[x][y] = 0.0001
    return n

class NeuralNetwork():

    def __init__(self, neurons):
        self.neurons = neurons
        self.weights = []
        self.bias = []
        self.fitness = 0
        previous_neuron = neurons[0]
        for neuron in neurons[1:]:
            weight = np.random.uniform(low = -2, high= 2,size=(previous_neuron, neuron))
            bia = np.random.uniform(size=(1, neuron))
            self.weights.append(weight)
            self.bias.append(bia)
            previous_neuron = neuron

    def setNN(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, data_input, function_activation):
        predictions = []
        for i in range(len(self.weights)):
            prediction = function_activation(data_input.dot(self.weights[i]) + self.bias[i])
            predictions.append(prediction)
            data_input = prediction
        return predictions

    def crossover(self, partner, mutation_rate):
        child0_weights = []
        child1_weights = []
        for i in range(len(self.weights)):
            weights = crossoverMatrix(self.weights[i], partner.weights[i])
            child0_weights.append(mutateMatrix(weights[0], mutation_rate))
            child1_weights.append(mutateMatrix(weights[1], mutation_rate))
        #bias
        child0_bias = []
        child1_bias = []
        for i in range(len(self.bias)):
            bias = crossoverMatrix(self.bias[i], partner.bias[i])
            child0_bias.append(mutateMatrix(bias[0], mutation_rate))
            child1_bias.append(mutateMatrix(bias[1], mutation_rate))
        nn0 = NeuralNetwork(self.neurons)
        nn1 = NeuralNetwork(self.neurons)
        nn0.setNN(child0_weights, child0_bias)
        nn1.setNN(child1_weights, child1_bias)
        return [nn0, nn1]

    def setFitness(self, args, function):
        self.fitness = function(args)

    def getFitness(self):
        return self.fitness
    def getInverseFitness(self, maxFitness):
        return maxFitness+1 - self.getFitness()
        

class GeneticAlgorithm():
    def __init__(self, n_pop, neurons):
        self.n_pop = n_pop
        self.population = [NeuralNetwork(neurons) for _ in range(n_pop)]

    def getFitness(self):
        fitness = []
        for nn in self.population:
            fitness.append(nn.getFitness())
        return fitness
    
    def getMeanFitness(self):
        return np.mean(self.getFitness())

    def getMaxFitness(self):
        return max(self.getFitness())

    def inverse_fitness(self):
        inverse_fitness = []
        for population in self.population:
            inverse_fitness.append(self.getMaxFitness()+1 - population.getFitness())
        return inverse_fitness

    def remove_pop(self, n):
        while n > 0:
            maxFitness = self.getMaxFitness()
            azaraduh = Roleta.pior([[nn.getInverseFitness(maxFitness), nn] for nn in self.population])
            self.population.remove(azaraduh)
            n -= 1

    def crossover(self, mutation_rate):
        childs = []
        i = 0
        while i < int(self.n_pop/4):
            dad = Roleta.torneio([[nn.getFitness(), nn] for nn in self.population], limiter = len(self.population)//8)#
            mom = Roleta.torneio([[nn.getFitness(), nn] for nn in self.population])
            while mom == dad:
                mom = Roleta.torneio([[nn.getFitness(), nn] for nn in self.population])
            babies = dad.crossover(mom, mutation_rate)
            childs.append(babies[0])
            childs.append(babies[1])
            i += 1
        self.remove_pop(i*2)
        for child in childs:
            self.population.append(child)

