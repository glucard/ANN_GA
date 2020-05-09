
import pygame
import numpy as np


def draw(screen, matrix, size):
    matrix.reverse()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(int(100*j-size/8), int(size*i), int(size/6), int(size*0.8)))
            if matrix[i][j] == 1:
                pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(int(100*j+size*0.3), int(size*i+size*0.1), int(size*0.3), int(size*0.5)))
            elif matrix[i][j] == 2:
                pygame.draw.circle(screen, (0, 255, 0), (int(100*j+size*0.45), int(size*i+size*0.1)), int(size*0.3))
    matrix.reverse()

def drawNN(screen, inputs, nn, size, activation_function):
    for i in range(len(nn.weights)):
        for j in range(len(nn.weights[i])):
            for k in range(len(nn.weights[i][j])):
                if nn.weights[i][j][k] >= 0:
                    r = int(255 * nn.weights[i][j][k]//2)
                    gb = int(100 - nn.weights[i][j][k] * 100//2)
                    color = (r, gb, gb)
                else:
                    b = int(-255 * nn.weights[i][j][k]//2)
                    rg = int(100 + nn.weights[i][j][k] * 100//2)
                    color = (rg, rg, b)
                pygame.draw.line(screen, color, (350+100*i, 300-50*j + 50*len(nn.weights[i])/2-25), (450+100*i, 300-50*k+ 50*len(nn.weights[i][j])/2-25))
    for i in range(len(inputs[0])):
        if inputs[0][i] == 0:
            color = (100,100,100)
        else:
            color = (255,0,0)
        pygame.draw.circle(screen, color, (350, int(300-50*len(inputs[0])/2+50*i+25)), 20)
    predictions = nn.predict(inputs, activation_function)
    for i in range(len(predictions)):
        minimo = min(predictions[i][0])
        maximo = max(predictions[i][0])
        for j in range(len(predictions[i][0])):
            try:
                r = int(255 * (predictions[i][0][j]-minimo) / (maximo-minimo))
                gb = int(100 - (predictions[i][0][j]-minimo) / (maximo-minimo) * 100)
            except:
                r = 100
                gb = 100
            color = (r,gb,gb)
            pygame.draw.circle(screen, color, (450+100*i , int(300-50*len(predictions[i][0])/2+50*j+25)), 20)
    