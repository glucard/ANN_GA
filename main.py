import cargame as cg
import pygame, time
import annga, drawCargame
import numpy as np
import random as rd
from operator import itemgetter

def ReLU(x, derivative = False):
    if(derivative):
        return 1. * (x > 0)
    return np.maximum(0,x)

def sigmoid (x, derivative = False):
    if(derivative):
        return x * (1 - x)
    return 1/(1 + np.exp(-x))

def fitnessFunction(args):
    desvios = args[0]
    metros = args[1]
    return desvios*5+metros

def roadToInput(jogo):
    input_frame = [0,0,0]
    input_frame[jogo.car.getX()] = 1
    for faixa in jogo.rua[1:2]:
        for n in faixa:
            input_frame.append(n)
    return np.array([input_frame])

#Creating ANNGA
GA = annga.GeneticAlgorithm(200, [6,12,12,3])
activation_function = ReLU
epochs = 1000
gen = 0

#Setting up pygame
screen = pygame.display.set_mode((900, 625))
done = False
while not done:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    
    #Training ANNGA
    if epochs > 0:
        print(epochs)
        max_fitness = GA.getMaxFitness()
        print("fitness =",max_fitness)
        epochs-=1
        for nn in GA.population:
            #Setting up cargame
            jogo = cg.CarGame(3,6)
            jogo.startCar()
            desvios = 0
            metros = 0
            while(jogo.update()):
                #PLAYING
                input_frame = roadToInput(jogo)
                if(nn.getFitness() == max_fitness and gen != 0): #if the best
                    #SHOW IN PYGAME
                    screen.fill((0, 0, 0))
                    drawCargame.draw(screen, jogo.rua,100)
                    drawCargame.drawNN(screen, input_frame, nn, 100, activation_function)
                    pygame.display.flip()
                    time.sleep(0.2)
                pygame.event.pump()
                metros+=1
                if True == jogo.moveIA(nn.predict(input_frame, activation_function)[-1]):
                    desvios += 1
                if metros > 500:
                    break
            nn.setFitness([desvios, metros], fitnessFunction)
        GA.crossover(2)
        gen += 1
