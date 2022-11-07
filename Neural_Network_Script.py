import numpy as np
import pandas as pd
import one_hot
import sys
import math


def SigmoidActivationFunction(weightedInput):
    return 1./(1. + np.exp(-weightedInput))

def SigmoidActivationFunctionDerivative(weightedInput):
    activation = SigmoidActivationFunction(weightedInput)
    
    return activation*(1. - activation)

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
        self.x = X
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
        #return error*error

    
    def NodeCostDerivative(self,outputActivation,expectedOutput):
        
        #return 2 * (np.subtract(outputActivation,expectedOutput))
        return  (np.subtract(outputActivation,expectedOutput))
    
    def ApplyGradients(self,eta,l2):

        self.bias_values -= self.costGradientB
        self.weights -= self.costGradientW * eta# + l2*self.weights

    #expected outputs to jeden wiersz [0 1 0] z y 
    def CalculateOutputLayerNodeValues(self,expectedOutputs):
        

        costDerivative = self.NodeCostDerivative(self.a, expectedOutputs)
        
        activationDerivative = SigmoidActivationFunctionDerivative(self.z)
        

        nodeValues = np.multiply(costDerivative,activationDerivative)
        #nodeValues = np.dot(activationDerivative,costDerivative.T)
        
        return nodeValues

    def CalculateHiddenLayerNodeValues(self,previousLayer,previousNodeValues):
        weightedInputDerivative = previousLayer.weights
        newNodeValues= np.dot(previousNodeValues,weightedInputDerivative.T)

        newNodeValues = np.multiply(newNodeValues,SigmoidActivationFunctionDerivative(self.z))

        return newNodeValues
        
    def UpdateGradient(self, nodeValues):

        derivativeCost_Weights = np.dot(self.x.T,nodeValues)

        self.costGradientW = derivativeCost_Weights

        derivativeCost_Bias = 1*nodeValues

        self.costGradientB = np.sum(derivativeCost_Bias,axis=0)/len(self.x)


    #def UpdateGradients(nodeValues):

    
    

class NeuralNetwork:
    def __init__(self,layerSizes,epoch,eta,l2,minibatch_size) -> None:
        self.layers = [None]*(len(layerSizes)-1)
        for i in range(len(self.layers)):
            self.layers[i] = Layer(layerSizes[i],layerSizes[i+1])

        self.l2 = l2
        self.epoch = epoch
        self.eta = eta
        self.errors = []
        self.minibatch_size = minibatch_size

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

    def UpdateAllGradients(self,X,y):
        
        self.network_forward_propagation(X)
        output_layer = self.layers[-1]
        nodeValues = output_layer.CalculateOutputLayerNodeValues(y)
        output_layer.UpdateGradient(nodeValues)

        hiddenLayerIdx = len(self.layers) -2
        while(hiddenLayerIdx>=0):
            hiddenLayer = self.layers[hiddenLayerIdx]
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(self.layers[hiddenLayerIdx+1],nodeValues)
            hiddenLayer.UpdateGradient(nodeValues)
            hiddenLayerIdx -= 1
    def ApplyAllGradients(self):

        for layer in self.layers:
            layer.ApplyGradients(self.eta,self.l2)

    def learn(self,X,y,X_test,y_test):

        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_eval = y
        y = one_hot._onehot(y,self.layers[len(self.layers)-1].outputNodes)
        
        for i in range(self.epoch):
            indices = np.arange(X.shape[0])

            if 1:
                np.random.RandomState(1).shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                self.UpdateAllGradients(X[batch_idx],y[batch_idx])
                self.ApplyAllGradients()

            #if i%10 == 0:
            cost = self.network_loss(X,y)
            self.errors.append(cost)
            y_train_pred = self.network_predict(X)
            y_valid_pred = self.network_predict(X_test)
            train_acc = ((np.sum(y_eval == y_train_pred)).astype(np.float) /
                         X.shape[0])
            valid_acc = ((np.sum(y_test == y_valid_pred)).astype(np.float) /
                         X_test.shape[0])
            sys.stderr.write('\r%0*d/%d | Koszt: %.2f '
                             '| Dokładność uczenia/walidacji: %.2f%%/%.2f%% ' %
                             (len(str(self.epoch)), i+1, self.epoch, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)