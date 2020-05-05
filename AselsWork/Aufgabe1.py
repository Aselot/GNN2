#Aygün Selvi
#Patrick Henschel

import numpy as np
import matplotlib.pyplot as plt


t =0.01
x1 = -7
x2=-0.2
x3=8



def EulerFunction(StartX,Counter,t):
    x=StartX
    xNeu = x+t*(x-np.power(x,3))
    arr = np.array(xNeu)
    for i in np.arange(0,Counter,1):
        x =xNeu
        xNeu = x+t*(x-np.power(x,3))
        arr = np.append(arr,xNeu)
    return arr

def Fixpoint(start,end):
    arr = np.empty()
    for i in range(start,end):
        y= -1/2*np.power(i,2)+1/4*np.power(i,2)
        np.append(arr, y)
    return arr


arr = EulerFunction(x1,600,t)
plt.title("-7")
plt.plot(arr)
plt.show()
arr = EulerFunction(x2,600,t)
plt.title("-0,2")
plt.plot(arr)
plt.show()
arr = EulerFunction(x3,600,t)
plt.title("8")
plt.plot(arr)
plt.show()


#Antworten: die Anfangspunkte laufen zu den nächstmöglichen Schnittpunkt, welche
# fixe punkten darstellen
# , fürr x1=-7 ist das x=-1
#für x2=-0,2 ist dies ebenfalls x=-1, und für x3=8 ist dies x=1,