import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from sklearn.datasets import make_moons, make_circles

#aquí pasamos la función sigmoide la cúal se utilizará para crear la red neuronal
def sigmoid(x, deriv = False):
    '''Si deriv es verdadero se retorna la derivada de la función sigmoide'''
    if deriv:
        return x * (1-x)
    return 1/(1+np.e**-x) #baias

def tanh(x, deriv = False):
    if deriv:
        return (1 - (np.tanh(x))**2)
    return np.tanh(x)

'''Se define la función del error cuadratico medio
    Aquí se calcula el error entre el valor que predice la red neuronal y el valor teorico
    que queremos que la red neuronal aprenda a realizar'''
def MSE(Yp, Yr, deriv = False): #Yp=valor predicho, Yr=Valor real
    if deriv:
        return (Yp - Yr)
    return np.mean((Yp-Yr)**2)

#Capas de la red neuronal
class Layer:
    # Aquí pasamos el numero de conexiones y el numero de neuronas con y neuron
    #se realizan productos matricialesde la capa anterior por la capa que sigue
    def __init__(self, con, neuron):
        self.b = np.random.rand(1, neuron) * 2 - 1 #Aquí se generan numeros aleatorios entre el 0 y el 1
        self.W = np.random.rand(con, neuron) * 2 - 1 #Las conexiones de la capa anterior por el numero de neuronas

#Aquí se guardaran las capas de la red neuronal
class NeuralNetwork:
    # top es la topología donde se especifica la cantidad de entradas y salidas de la red neuronal
    def __init__(self, top = [], act_fun = sigmoid):
        self.top = top
        self.act_fun = act_fun
        self.model = self.define_model()

    #Está función genera la red neuronal en el atributo model
    def define_model(self):

        NeuralNetwork = [] #Lista de objetos tipo Layer
        # Aquí se agregan objetos layer al array anterior
        for i in range(len(self.top[:-1])):
            #Aquí se hace la comparación de las neuronas de la capa anterior con la capa actual
            NeuralNetwork.append(Layer(self.top[i], self.top[i+1]))
        return NeuralNetwork
    #se le pasan los datos para realizar las predicciones
    def predict(self, X = []):

        out = X
        #Aquí recorremos nuestra red neuronal self.model
        for i in range(len(self.model)):
            #Inicializamos una variable que servirá como salida de cada capa
            #Después pasamos esos valores a la función de activación act_fun realizando el producto matricial
            z = self.act_fun( out @ self.model[i].W + self.model[i].b )
            #Ahora asignamos z a la variable out porque la entrada del primer Layer va a ser igual a la salida del Layer anterior
            #Esto se hace para que out valga lo que es Z durante cada recorrido
            out = z

        return out

    #A continuación inicializamos el método fit, el cual se encargará de entrenar la red neuronal
    '''X = Conjunto de datos, Y = Observaciones reales para aprender, 
    El leaning_rate nos dice que tanto nos vamos a mover en el espacio de error
    Si tenemos un learning rate muy alto, a la red neuronal le costará mucho converger a un resultado
    Mientras que si es muy bajo, tiende a aprender de forma muy lenta'''
    def fit(self, X = [], Y = [], epochs = 100, learning_rate = 0.25):

        for k in range(epochs):
            #Creamos una lista de tuplas, donde en el None guardaremos la función de activación (Producto matricial)
            #En X Guardaremos la función de la salida de activación
            out = [(None, X)]

            for i in range(len(self.model)):
                z = out[-1][1] @ self.model[i].W + self.model[i].b #Hacemos la combinación lineal
                a = sigmoid(z, deriv = False) #Aquí se guarda la salida
                #Agrego a la tupla el valor de la combinación lineal y la salida
                out.append((z, a)) #Esto nos ayuda a ver el error de cada Layer (Backpropagation)
            #Se guardan los valores de la función de coste
            deltas = []
            #Aquí se recorren las capas en sentido contrario (Reversa)
            for i in reversed(range(len(self.model))):
                #Guardamos en cada iteración el z de la última capa de activación
                z = out[i + 1][0]
                a = out[i + 1][1]

                #Está parte sirve para que no salte un error porque el deltas no tiene ningún valor agregado inicialmente
                if i == len(self.model) - 1:
                    #Agregamos que tanto se equivocó nuestro modelo en cada Layer
                    deltas.insert(0, MSE(a, Y, deriv = True) * sigmoid(a, deriv = True))
                else:

                    deltas.insert(0, deltas[0] @ _W.T * sigmoid(a, deriv = True))

                _W = self.model[i].W

                #Ahora pasamos a actualizar las matrices
                #Aquí se calcula el vector promedio de lo que se obtuvo en el array deltas y eso se lo restamos a la matriz principal
                #Esto significa que se va a mover hacia el lado donde decrece el gradiante
                self.model[i].b = self.model[i].b - np.mean(deltas[0], axis = 0, keepdims = True) * learning_rate
                #En función de cada Layer, estamos corrigiendo la matriz con la salida que tuvo, multiplicandola por el vector deltas
                #Aquí el Learning Rate nos minimiza el error
                self.model[i].W = self.model[i].W - out[i][1].T @ deltas[0] * learning_rate

        print('RED NEURONAL ENTRENADA CON ÉXITO')

#Está función sirve para generar puntos aleatorios
def random_points(n = 100):
	x = np.random.uniform(0.0, 1.0, n)
	y = np.random.uniform(0.0, 1.0, n)

	return np.array([x, y]).T

def main():
    raiz = Tk()
    raiz.title("Redes Neuronales Con Phyton")
    # Dimensiones de la ventana y ubicación en pantalla
    raiz.geometry("600x600+300+60")
    raiz.config(background="lightyellow")

    def Xoor():
        brain_xor = NeuralNetwork(top=[2, 4, 1], act_fun=tanh)

        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])

        Y = np.array([
            [0],
            [1],
            [1],
            [0],
        ])

        brain_xor.fit(X=X, Y=Y, epochs=10000, learning_rate=0.08)

        x_test = random_points(n=5000)
        y_test = brain_xor.predict(x_test)

        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=25, cmap='GnBu')
        # plt.savefig('redis_neuron.jpg')
        plt.show()
    def circle():
        brain_circles = NeuralNetwork(top=[2, 4, 8, 1], act_fun=sigmoid)

        X, Y = make_circles(n_samples=500, noise=0.05, factor=0.5)
        Y = Y.reshape(len(X), 1)

        brain_circles.fit(X=X, Y=Y, epochs=10000, learning_rate=0.05)

        y_test = brain_circles.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_test, cmap='winter', s=25)
        # plt.savefig('Circle_neuron.jpg')
        plt.show()
    def moons():
        brain_moon = NeuralNetwork(top=[2, 4, 8, 1], act_fun=sigmoid)

        X, Y = make_moons(n_samples=500, noise=0.05)
        Y = Y.reshape(len(X), 1)

        y_test = brain_moon.predict(X)

        brain_moon.fit(X=X, Y=Y, epochs=10000, learning_rate=0.05)

        y_test = brain_moon.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_test, cmap='winter')
        # plt.savefig('Moon_neuron.jpg')
        plt.show()

    #print("\tBienvenidos a las redes neuronales\n")
    #desicion = int(input(print("Escriba a continuación que red neuronal desea ver:\n1. Xor\n2. Circulos\n3. Lunas\n")))

    #----------------------------------------Interfaz Gráfica---------------------------------------------
    # Título
    Titu = Label(raiz, text="BIENVENIDO A LAS REDES NEURONALES", font=("Conic Sans MS", 13, "bold"))
    Titu.place(x=125, y=100)
    # Mensaje
    Titu = Label(raiz, text="Seleccione a continuación que red neuronal que desea ver", font=("Conic Sans MS", 13))
    Titu.place(x=75, y=200)
    # Boton para llamar al ejemplo Xoor
    botXor = Button(raiz, text="Ejemplo Xor", command=Xoor)
    botXor.pack()
    botXor.place(x=150, y=350)
    # Boton para llamar al ejemplo Circulos
    botCircle = Button(raiz, text="Ejemplo Círculo", command=circle)
    botCircle.pack()
    botCircle.place(x=350, y=350)
    # Boton para llamar al ejemplo Lunas
    botMoon = Button(raiz, text="Ejemplo Lunas", command=moons)
    botMoon.pack()
    botMoon.place(x=250, y=450)

    raiz.resizable(False, False)

    raiz.mainloop()

if __name__ == '__main__':
    main()