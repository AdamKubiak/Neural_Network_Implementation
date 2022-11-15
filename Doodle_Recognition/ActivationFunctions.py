import numpy as np


class Activation:

    def SigmoidActivationFunction(self,weightedInput):
        return 1./(1. + np.exp(-weightedInput))

    def SigmoidActivationFunctionDerivative(self,weightedInput):
        activation = 1./(1. + np.exp(-weightedInput))

        return activation*(1. - activation)

    def TanHActivationFunction(self,weightedInput):
        e2 = np.exp(2 * weightedInput)
        return (e2 - 1) / (e2 + 1)

    def TanHActivationFunctionDerivative(self,weightedInput):
        e2 = np.exp(2 * weightedInput)
        t = (e2 - 1) / (e2 + 1)
        return 1 - t * t

    def ReLUActivationFunction(self,weightedInput):
        return np.maximum(0, weightedInput)

    def ReLUActivationFunctionDerivative(self,weightedInput):
        array = np.where(weightedInput > 0, 1, 0)

        return array

    def SoftmaxActivationFunction(self,weightedInput):
        z = np.max(weightedInput, axis=1)
        weightedInput = weightedInput - z[:,None]
        e = np.exp(weightedInput)/np.sum(np.exp(weightedInput), axis=1, keepdims=True)

        return e

    def SoftmaxActivationFunctionDerivative(self,weightedInput):
        z = np.max(weightedInput,axis=1)
        weightedInput = weightedInput - z[:,None]
        ex = np.exp(weightedInput)
        expSum = np.sum(np.exp(weightedInput),  axis=1, keepdims=True)
        return (ex*expSum-ex*ex)/(expSum*expSum)

    def ActivationFunctionPick(self,name):

        if name == 'Sigmoid':
            return self.SigmoidActivationFunction

        if name == 'TanH':
            return self.TanHActivationFunction

        if name == 'ReLU':
            return self.ReLUActivationFunction

        if name == 'Softmax':
            return self.SoftmaxActivationFunction

    def ActivationFunctionDerivativePick(self,name):

        if name == 'Sigmoid':
            return self.SigmoidActivationFunctionDerivative

        if name == 'TanH':
            return self.TanHActivationFunctionDerivative

        if name == 'ReLU':
            return self.ReLUActivationFunctionDerivative

        if name == 'Softmax':
            return self.SoftmaxActivationFunctionDerivative

        return 0
