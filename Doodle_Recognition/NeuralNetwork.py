import Layer as lyr
import one_hot
import sys
import numpy as np


class NeuralNetwork:
    def __init__(self,layerSizes,epoch,eta,l2,minibatch_size,HiddenLayerActivation,OutputLayerActivation) -> None:
        self.layers = [None]*(len(layerSizes)-1)
        for i in range(len(self.layers)):
            if i == (len(self.layers)-1):
                self.layers[i] = lyr.Layer(layerSizes[i],layerSizes[i+1],OutputLayerActivation)
            else:
                self.layers[i] = lyr.Layer(layerSizes[i],layerSizes[i+1],HiddenLayerActivation)

        self.l2 = l2
        self.epoch = epoch
        self.eta = eta
        self.errors = []
        self.minibatch_size = minibatch_size

    """
    X ---> zbiór danych uczących

    Metoda przeprowdzana propagację w przód dla każdej z wartw wykorzystując wartosci aktywacji z poprzedniej aktywacji
    w celu wykonania obliczeń dla następnej warstwy.
    Wyjściem są wartości aktywacji ostatniej wartstwy sieci neuronowej
    """
    def network_forward_propagation(self,X):
        for layer in self.layers:
            X = layer.forward_propagation(X)

        return X

    """
    Metoda dokonuje klasyfikacji danych przy pomocy wartości aktywacji ostatnich neuronów z ostatniej warstwy i zwraca indeks neuronu o najwyższej wartości
    Zwracany jest indeks ponieważ etykiety są przedstawione w formacie one_hot
    """
    def network_predict(self,X):
        activation_values = self.network_forward_propagation(X)

        predictions = np.argmax(activation_values,axis = 1)

        return predictions
    
    """
    Metoda strat. Im większa wartość strat tym gorzej sieć radzi sobie z klasyfikowaniem danych 
    """
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

    """
    Metoda dokonuje obliczeń algorytmu wstecznej propagacji
    """
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

    """
    Metoda aplikuje wyliczoną wartość gradientu dla każdej z warstw sieci neuronowej
    """
    def ApplyAllGradients(self):

        for layer in self.layers:
            layer.ApplyGradients(self.eta,self.l2)

    
    """
    Metoda uczenia sieci neuronowej
    """
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

            if i%1 == 0:
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