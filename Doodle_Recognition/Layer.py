import numpy as np
import ActivationFunctions as af

class Layer:
    def __init__(self,inputNodes,outputNodes,ActivatioName) -> None:

        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        random = np.random.RandomState(1)
        afObject = af.Activation()
        self.activation = afObject.ActivationFunctionPick(ActivatioName)
        self.activationDerivative = afObject.ActivationFunctionDerivativePick(ActivatioName)
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
    Metoda dokonuje operacji propagacji w przód dla obiektu Layer
    X - wejścia warstwy

    przemnożone wejścia przez odpowiednie wagi które są połączeniami neuronów i dodane do nich wartości bias
    zostają przepuszczone przez funkcję aktywacji.

    Wyjściem są wejścia dla kolejnej warstwy sieci neuronowej
    """
    def forward_propagation(self,X):
        self.x = X
        output_values = np.dot(X,self.weights) + self.bias_values
        self.z = output_values
        activation_values = self.activation(output_values)
        print(activation_values.shape)
        #activation_values = SigmoidActivationFunction(output_values)
        self.a = activation_values
        return activation_values

    """
    X: tablica [n_próbek,n_cech]
    -> tablica [n_próbek]
    Metoda nie jest wykorzystywana w sieci neuronowej
    """
    def predict(self,X):
        activation_values = self.forward_propagation(X)

        preditions = np.argmax(activation_values,axis=1)

        return preditions
    

    """
    Metoda kosztu. Im większa wartość kosztu tym gorzej sieć radzi sobie z klasyfikowaniem danych 
    """
    def NodeCost(self,outputActivation,expectedOutput):
        error = outputActivation - expectedOutput

        return 0.5*error*error
        #return error*error

    """Pochodna Metoda kosztu"""
    def NodeCostDerivative(self,outputActivation,expectedOutput):
        
        #return 2 * (np.subtract(outputActivation,expectedOutput))
        return  (np.subtract(outputActivation,expectedOutput))
    
    """
    Każda z warstw sieci posiada macierze costGradientB i costGradientW w których zapisywane są poprawki jakie należy wprowadzić do wag 
    danej warstwy. Po obliczeniu gradientów podczas operacji wstecznej propagacji, wyliczone wartości są odejmowane od wag i biasów warstwy.
    """
    def ApplyGradients(self,eta,l2):

        self.bias_values -= self.costGradientB
        self.weights -= self.costGradientW * eta# + l2*self.weights

    #expected outputs to jeden wiersz [0 1 0] z y 
    """
    Metoda ma za zadanie wyliczyć nodeValues
    W programie określone zostają tak wartości wyliczane dla neuronów poprzez wykorzystanie pochodnych z metody łańcuchowej w celu przeprowadzenia wstecznej propagacji
    W tej motodzie zostają wyliczone nodeValues dla ostatniej warstwy sieci neurownowej
    """
    def CalculateOutputLayerNodeValues(self,expectedOutputs):
        

        costDerivative = self.NodeCostDerivative(self.a, expectedOutputs)
        
        #activationDerivative = SigmoidActivationFunctionDerivative(self.z)
        activationDerivative = self.activationDerivative(self.z)
        

        nodeValues = np.multiply(costDerivative,activationDerivative)
        #nodeValues = np.dot(activationDerivative,costDerivative.T)
        
        return nodeValues

    """
    Metoda ma za zadanie wyliczyć nodeValues
    W programie określone zostają tak wartości wyliczane dla neuronów poprzez wykorzystanie pochodnych z metody łańcuchowej w celu przeprowadzenia wstecznej propagacji
    W tej motodzie zostają wyliczone nodeValues dla ukrytych warstw sieci neurownowej
    """
    def CalculateHiddenLayerNodeValues(self,previousLayer,previousNodeValues):
        weightedInputDerivative = previousLayer.weights
        newNodeValues= np.dot(previousNodeValues,weightedInputDerivative.T)

        #newNodeValues = np.multiply(newNodeValues,SigmoidActivationFunctionDerivative(self.z))
        newNodeValues = np.multiply(newNodeValues,self.activationDerivative(self.z))

        return newNodeValues

    """ 
    Po obliczeniu wartości nodeValues możemy przeprowadzić uaktualnienie wartości gradientu z uwzględnieniem wejść zapisanych w kazdej z warstw podczas przeprowadzanej 
    propagacji w przód w parametrze 'self.x'
    """   
    def UpdateGradient(self, nodeValues):

        derivativeCost_Weights = np.dot(self.x.T,nodeValues)

        self.costGradientW = derivativeCost_Weights

        derivativeCost_Bias = 1*nodeValues

        self.costGradientB = np.sum(derivativeCost_Bias,axis=0)/float(len(self.x))