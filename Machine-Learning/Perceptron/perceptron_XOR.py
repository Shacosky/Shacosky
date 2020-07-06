import numpy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#el siguiente segmento de codigo se utuliza para ocultar las warnings y hacer este codigo mas claro
import warnings

warnings.filterwarnings('ignore')

def tanh(x):
    return (1.0 - numpy.exp(-2*x))/(1.0 + numpy.exp(-2*x))

def tahn_derivative(x):
    return (1+x)*(1-x)

class NeuralNetwork:
    def __init__(self, net_arch):
        numpy.random.seed(1)
        self.activity = tanh
        self.activity_derivate = tahn_derivative
        self.layers = len(net_arch)
        self.step_per_epoch = 1
        self.arch = net_arch
        self.weigths = []

        for layer in range(self.layers - 1):
            w = 2*numpy.random.rand(net_arch[layer] + 1, net_arch[layer+1])-1
            self.weigths.append(w)

    def _forward_prop(self, x):
        y = x

        for i in range(len(self.weigths)-1):
            activation = numpy.dot(y[i], self.weigths[i])
            activity = self.activity(activation)

            activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
            y.append(activity)

        #Ultima capa
        activation = numpy.dot(y[-1],self.weigths[-1])
        activity = self.activity(activation)
        y.append(activity)

        return y

    def _back_prop(self, y, target, learning_rate):
        error = target -y[-1]
        delta_vec = [error * self.activity_derivate(y[-1])]

        for i in range(self.layers-2,0,-1):
            error = delta_vec[-1].dot(self.weigths[i][1:].T)
            error = error* self.activity_derivate(y[i][1:])
            delta_vec.append(error)

        delta_vec.reverse()        

        for i in range(len(self.weigths)):
            layer = y[i].reshape(1,self.arch[i]+1)
            delta = delta_vec[i].reshape(1, self.arch[i+1])
            self.weigths[i] += learning_rate*layer.T.dot(delta)
    
    def fit(self, data, labels, learning_rate=0.1, epochs=100):
        ones = numpy.ones((1,data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)

        for k in range(epochs):
            if(k+1) % 10000 == 0:
                print('epochs: {}'.format(k+1))

            sample=numpy.random.randint(X.shape[0])

            x = [Z[sample]]
            y=self._forward_prop(x)
            target = labels[sample]
            self._back_prop(y, target, learning_rate)

    def predict_single_data(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0,len(self.weigths)):
            val = self.activity(numpy.dot(val,self.weigths[i]))
            val = numpy.concatenate((numpy.ones(1).T,numpy.array(val)))
        return val[1]

    def predict(self, X):
        Y = numpy.array([]).reshape(0,self.arch[-1])
        for x in X:
            y = numpy.array[[self.predict_single_data(x)]]
            Y = numpy.vstack((Y,y))
        return Y

numpy.random.seed(0)

nn = NeuralNetwork([4,4,1])

X = numpy.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1],
])

y = numpy.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]) #XOR

nn.fit(X,y,epochs=100000)

print("Prediccion FInal de los valores de salida")

errores = []
for s in X:
    print(s,nn.predict_single_data(s))
    errores.append(nn.predict_single_data(s))

print(str(len(errores)) + "-> Numero de veces que se itero el peso")
print(str(errores)+ "-> numero de iteraciones")

plt.plot(range(1, len(errores) + 1), errores, marker='o')
plt.xlabel('Epochs')
plt.ylabel('errores')
plt.show()