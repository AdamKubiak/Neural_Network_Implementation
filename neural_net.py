from typing_extensions import Self
import numpy as np
import pandas as pd
import one_hot
import ActivationFuntions

class Layer:
    def __init__(self,nInputNeurons,nOutputNeurons) -> None:
        #weights[wiersze,kolumny]
        #pierwszy wiersz to biasy
        self.nInputNeurons = nInputNeurons
        self.nOutputNeurons = nOutputNeurons
        self.eta = 0.01
        self.n_iter = 50
        self.weights = np.random.rand(nInputNeurons+1, nOutputNeurons) - 0.5

    def forward_propagation(self,X):
        """Wyliczenie propagacji w przód

        """

    def fit(self,X,y):
        for _ in range(self.n_iter):
            for xi,target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                
                print(update)

    def  NeuronCost(self,activationValue,expectedValue):
        error = activationValue - expectedValue

        return math.pow(error,2)

    def CalculateOutputs(self,input_data):
        activationValues = [0]*self.nOutputNeurons

        for i in range(len(activationValues)):
            w_ = self.weights[:,i]
            activationValues[i] = SigmoidActivationFunction(np.dot(input_data,w_[1:]) + w_[0])
        
        #print(weightedInputs)
        return activationValues

    def predict(self,input_data):
        output = self.CalculateOutputs(input_data)

        #return output.index(max(output))
        return np.where(output[0] > output[1],0,1)




class NeuralNetwork:
    def __init__(self,layerSizes) -> None:
        self.layers = [None]*(len(layerSizes)-1)
        #print(len(self.layers))
        for i in range(len(self.layers)):
            self.layers[i] = Layer(layerSizes[i],layerSizes[i+1])
        


    
    def CalculateOutputs(self,input_data):
        for layer in self.layers:
            input_data = layer.CalculateOutputs(input_data)
            

        return input_data

    def Classify(self,input_data):
        outputs = self.CalculateOutputs(input_data)
        print(outputs)
        return outputs.index(max(outputs))

    def CostFuntion(self,input_data,y):
        outputs = self.CalculateOutputs(input_data)
        outputLayer = self.layers[len(self.layers-1)]
        cost = 0.0
        for i in range(len(outputs)):
            cost += outputLayer.NeuronCost(outputs[i], y)





layer = Layer(2,2)




#print(layer.weights[0,1])
#print(layer.weights[:,0])

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

y =  df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',0,1)
y = one_hot._onehot(y,2)
#one_hot.to_OneHot(y)
print(y)
X = df.iloc[0:100,[0,2]].values

#layer.fit(X,y)
layers = [2,4,2]
network = NeuralNetwork(layers)
print(network.Classify(X[0]))
print(layer.CalculateOutputs(X[0]))
#network.Classify(X)
"""
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # rysuje wykres próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


from matplotlib.widgets import Slider, Button
plot_decision_regions(X, y, classifier=layer)
axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
axamplitude = plt.axes([0.25, 0.1, 0.65, 0.03])
freq = Slider(axfreq, 'Frequency', 0.0, 20.0, 3)
 
# Create a slider from 0.0 to 10.0 in axes axfreq
# with 5 as initial value and valsteps of 1.0
amplitude = Slider(axamplitude, 'Amplitude', 0.0,
                   10.0, 5, valstep=1.0)
freq.on_changed(plot_decision_regions)
amplitude.on_changed(plot_decision_regions)
plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')


#plt.savefig('rysunki/02_08.png', dpi=300)
plt.show()
"""