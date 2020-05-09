import random as rd

class Tree():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
        
    def getY(self):
        return self.y

    def update(self):
        self.y -= 1

class Car():
    def __init__(self, x = 1, y = 0):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
        
    def getY(self):
        return self.y

    def mover(self, movimento):
        if movimento == [[1],[0],[0]]:
            self.x -= 1
            #print("AQUI0")
        elif movimento == [[0],[0],[1]]:
            self.x += 1
            #print("AQUI1")
        else:
            pass

class CarGame():
    def __init__(self, faixa, distancia):
        self.faixa = faixa
        self.distancia = distancia
        self.rua = []
        self.trees = []
        self.car = None
        for _ in range(self.distancia):
            faixa = []
            for _ in range(self.faixa):
                faixa.append(0)
            self.rua.append(faixa)

    def getFaixa(self):
        return self.faixa
        
    def getDistancia(self):
        return self.distancia

    def createTree(self):
        tree = Tree(rd.randrange(self.getFaixa()), self.getDistancia())
        self.trees.append(tree)

    def isClearRoad(self, x, y):
        if(x == self.car.getX() and y == self.car.getY):
            return False
        return True

    def updateRoad(self):
        for y in range(self.distancia):
            for x in range(self.faixa):
                    self.rua[y][x] = 0

    def updateTrees(self):
        removeTrees = []
        for tree in self.trees:
            tree.update()
            if(tree.getY() >= 0):
                self.rua[tree.getY()][tree.getX()] = 2
            elif(tree.getY() == 0):
                pass
            else:
                removeTrees.append(tree)
        for tree in removeTrees:
            self.trees.remove(tree)
    
    def notClearCar(self):
        for tree in self.trees:
            if (self.car.getX() == tree.getX() and self.car.getY() == tree.getY()):
                return True
        return False

    def updateCar(self):
        if(self.car.getX() < 0):
            self.car.x = 0
        elif(self.car.getX() > 2):
            self.car.x = 2
        if self.notClearCar():
            raise ValueError("BATEU NA ARVOREEEE FIAUMMMM")
        self.rua[self.car.getY()][self.car.getX()] = 1

    def startCar(self):
        self.car = Car()
        self.updateCar()

    def mover(self, movimento):
        self.car.mover(movimento)        

    def moveIA(self, movimento):
        #print(movimento)
        ponto = False
        self.car.y = 1
        if(self.notClearCar()):
            ponto = True
        self.car.y = 0
        if movimento[0][0] > movimento[0][1] and movimento[0][0] > movimento[0][2]:
            if self.car.x == 0:
                ponto = False
            a = 1
            w = 0
            d = 0
        elif movimento[0][1] > movimento[0][0] and movimento[0][1] > movimento[0][2]:
            a = 0
            w = 1
            d = 0
            ponto = False 
        else:
            if self.car.x == 2:
                ponto = False
            a = 0
            w = 0
            d = 1
        self.mover([[a],[w],[d]])
        return ponto

    def moveByKeyboard(self):
        key = input("movimento: ")
        if key == 'a':
            movimento = [[1],[0],[0]]
        elif key == 'd':
            movimento = [[0],[0],[1]]
        else:
            movimento = [[0],[1],[0]]
        self.mover(movimento)

    def show(self):
        print("\n"*10)
        self.rua.reverse()
        for distancia in self.rua:
            for faixa in distancia:
                print("{}".format(faixa), end=" ")
            print("")
        self.rua.reverse()
    
    def update(self):
        self.createTree()
        self.updateRoad()
        self.updateTrees()
        try:
            self.updateCar()
        except:
            return False
        else:
            return True

#jogo = CarGame(3,6)
#jogo.startCar()
#jogo.show()
#while(jogo.update()):
#    jogo.show()
#    jogo.moveByKeyboard()
