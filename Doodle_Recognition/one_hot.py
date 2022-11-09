from sklearn.preprocessing import OneHotEncoder
import numpy as np
#FINAL
def _onehot(y, n_classes):
        """Koduje etykiety do postaci "gorącojedynkowej"

        Parametry
        ------------
        y : tablica, wymiary = [n_próbek]
            Wartości docelowe.

        Zwraca
        -----------
        onehot : tablica, wymiary = (n_etykiet, n_próbek)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.
        return onehot.T