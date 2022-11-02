import numpy as np
import pandas as pd
import one_hot
import math


def SigmoidActivationFunction(weightedInput):
    return 1/(1 + np.exp(-weightedInput))

class Layer:
    def __init__(self,inputNodes,outputNodes) -> None:

        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        random = np.random.RandomState(1)
        """
        WEIGHTS
        tworzy 2-wymiarową listę z wartościami wag 
        Przykładowo gdy warstwa ma 2 wejścia i 3 wyjścia
        [w11   w21   w31]
        [w12   w22   w32]
        [w13   w23   w33]
        Z wartości pierwszej kolumny obliczana jest wartość wyjściowa dla 1 wyjścia
        Z wartości drugiej kolmny obliczana jest wartość wyjściowa 2 wyjścia
        BIAS_VALUES
        tworzy 1-wymiarową listę o wymiarze ilości wyjść dla warstwy
        Do wartości wyliczonej z popagacji do przodu do wyjścia dodawana jest wartość bias
        3 wyjścia
        [bias1 bias2 bias3]
        """
        self.weights = random.normal(loc=0.0, scale=0.1, size=(inputNodes, outputNodes))
        self.bias_values = np.zeros(outputNodes)
        self.costGradientW = np.zeros(shape = (inputNodes,outputNodes))

        self.costGradientB = np.zeros(shape = (outputNodes))


    """
    Gdy mamy doczynienia z pierwszą warstwą sieci 2 cechy wejścia i 2 klasy wyścia(layer(2,2))
    funkcja przelicza cechy*wagi i zwraca wartości wyjściowych neurownów 
    Gdy wsadzimy w funkcję listę [n_próbek, n_cech] to zwraca [n_próbek, n_neuronów_wyjściowych]
    dla 5 próbek i sieci (2,2) [5,2]
    X : tablica [n_próbek, n_cech]
    -> [n_próbek, n_outputNodes]
    """
    def forward_propagation(self,X): 
        output_values = np.dot(X,self.weights) + self.bias_values
        self.z = output_values
        activation_values = SigmoidActivationFunction(output_values)
        self.a = activation_values
        return activation_values

    """
    X: tablica [n_próbek,n_cech]
    -> tablica [n_próbek]
    """
    def predict(self,X):
        activation_values = self.forward_propagation(X)

        preditions = np.argmax(activation_values,axis=1)

        return preditions

    def NodeCost(self,outputActivation,expectedOutput):
        error = outputActivation - expectedOutput

        return 0.5*error*error
    
    def ApplyGradients(self,eta):

        self.bias_values -= self.costGradientB
        self.weights -= self.costGradientW * eta

class NeuralNetwork:
    def __init__(self,layerSizes,epoch,eta) -> None:
        self.layers = [None]*(len(layerSizes)-1)
        for i in range(len(self.layers)):
            self.layers[i] = Layer(layerSizes[i],layerSizes[i+1])

        self.l2 = 0.
        self.epoch = epoch
        self.eta = eta

    def network_forward_propagation(self,X):
        for layer in self.layers:
            X = layer.forward_propagation(X)

        return X

    def network_predict(self,X):
        activation_values = self.network_forward_propagation(X)

        predictions = np.argmax(activation_values,axis = 1)

        return predictions
    
    def network_loss(self,X,y):
        activation_values = self.network_forward_propagation(X)
        
        outputLayer = self.layers[len(self.layers)-1]
        total_cost = 0
        cost_single_data = 0

        for xi, target in zip(activation_values,y):
            for i in range(len(activation_values[0])):
                cost_single_data += outputLayer.NodeCost(xi[i],target[i]) 

            total_cost += cost_single_data
            cost_single_data = 0

        return total_cost / len(X)

    #def ApplyAllGradients(self):


    def learn(self,X,y):

        h = 0.0001
        y = one_hot._onehot(y,self.layers[len(self.layers)-1].outputNodes)
        
        originalCost = self.network_loss(X,y)
        for i in range(self.epoch):
            for layer in self.layers:
                for nodeIn in range(layer.inputNodes):
                    for nodeOut in range(layer.outputNodes):
                        layer.weights[nodeIn][nodeOut] +=h
                        deltaCost = self.network_loss(X,y) - originalCost
                        layer.weights[nodeIn][nodeOut] -=h
                        layer.costGradientW[nodeIn][nodeOut] = deltaCost/h


                for biasIdx in range(layer.outputNodes):
                    layer.bias_values[biasIdx] +=h
                    deltaCost = self.network_loss(X,y) - originalCost
                    layer.bias_values[biasIdx] -=h
                    layer.costGradientB[biasIdx] = deltaCost/h
                print(layer.costGradientW.shape)
                print(layer.costGradientW)
                print(layer.costGradientB)
                layer.ApplyGradients(self.eta)

            print(self.network_loss(X,y))



        


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

y =  df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',0,1)
X = df.iloc[0:100,[0,2]].values




layers = [2,3,2]
network = NeuralNetwork(layers,10,0.01)
network.learn(X,y)

