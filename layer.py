import numpy as np
import pandas as pd
import one_hot
import math


def SigmoidActivationFunction(weightedInput):
    return 1/(1 + np.exp(-weightedInput))

def SigmoidActivationFunctionDerivative(sigmoid_input):
    return sigmoid_input(1-sigmoid_input)

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



    """
    Gdy mamy doczynienia z pierwszą warstwą sieci 2 cechy wejścia i 2 klasy wyścia(layer(2,2))
    funkcja przelicza cechy*wagi i zwraca wartości wyjściowych neurownów 

    Gdy wsadzimy w funkcję listę [n_próbek, n_cech] to zwraca [n_próbek, n_neuronów_wyjściowych]
    dla 5 próbek i sieci (2,2) [5,2]

    X : tablica [n_próbek, n_cech]
    -> [n_próbek, n_outputNodes]
    """
    def forward_propagation(self,X):
        
        #print(X)
        output_values = np.dot(X,self.weights) + self.bias_values
        self.z = output_values
        #print(output_values)
        activation_values = SigmoidActivationFunction(output_values)
        self.a = activation_values
        #print(activation_values)
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

        return error**2.


class NeuralNetwork:
    def __init__(self,layerSizes) -> None:
        self.layers = [None]*(len(layerSizes)-1)
        for i in range(len(self.layers)):
            self.layers[i] = Layer(layerSizes[i],layerSizes[i+1])
        
        self.l2 = 0.01
        self.epochs = 20
        self.eta = 0.0005
        self.minibatch_size = 10

    def network_forward_propagation(self,X):
        for layer in self.layers:
            X = layer.forward_propagation(X)
        
        #print(X)
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
                print(i)
                cost_single_data += outputLayer.NodeCost(xi[i],target[i])
            
            total_cost += cost_single_data
            cost = 0

        return total_cost / len(X)

    def network_logistic_cost(self,y,final_activation_values):
        L2_term = 0

        for layer in self.layers:
            L2_term += np.sum(layer.weights**2.)
        
        L2_term = L2_term*self.l2

        term1 = -y * (np.log(final_activation_values + 1e-5))
        term2 = (1. - y) * np.log(1. - final_activation_values + 1e-5)
        cost = np.sum(term1 - term2) + L2_term

        return cost
    
    def network_backward_propagation(self,X,y,batch_idx):

        sigma_out = self.network_forward_propagation(X[batch_idx]) - y[batch_idx]
        check = 0
        for layer in self.layers:
            sigmoid_derivative = layer.a*(1.-layer.a)

            # [n_próbek, n_etykiet_klas] dot [n_ etykiet_klas, n_hidden]
            # -> [n_ próbek, n_hidden]
            sigma = (np.dot(sigma_out,layer.weights.T) * sigmoid_derivative)
            if check == 0 :
                grad_weights = np.dot(X[batch_idx].T,sigma)
            else:
                grad_weights = np.dot(X.T,sigma)
            grad_bias = np.sum(sigma,axis = 0)
            X = layer.forward_propagation(X)

            #regularyzacja i aktualizowanie wag
            delta_weights = (grad_weights + self.l2*layer.weights)
            delta_bias = grad_bias

            layer.weights -= self.eta*delta_weights
            layer.bias_values -=self.eta*delta_bias
            check+=1

    
    def learn(self,X,y):

        y = one_hot._onehot(y, self.layers[len(self.layers)-1].outputNodes)


        for i in range(self.epochs):
            indices = np.arange(X.shape[0])

            if True:
                random = np.random.RandomState(1)
                random.shuffle(indices)

            for start_idx in range(0,indices.shape[0] - self.minibatch_size +1,self.minibatch_size):
                batch_idx = indices[start_idx:start_idx+self.minibatch_size]

                self.network_backward_propagation(X,y,batch_idx)
            
            #Ewaluacja

            cost = self.network_logistic_cost(y,network.network_forward_propagation(X))
            print(cost)


        return self


layers = [2,4,2]
network = NeuralNetwork(layers)

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

y =  df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',0,1)
#y = one_hot._onehot(y,2)
#print(y)
X = df.iloc[0:100,[0,2]].values
    
network.learn(X,y)

#print(layer.predict(X))
#print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#print(network.network_predict(X))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print(network.network_logistic_cost(y,network.network_forward_propagation(X)))


