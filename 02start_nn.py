import numpy as np
import matplotlib.pyplot as plt

def act(x):
    return 0 if x <= 0 else 1

def go(C):
    x = np.array([C[0], C[1], 1])
    print("Веса определяли из уравнений\n\n\n")
    print("получили на вход\n", x,"\n")
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    print("веса для 2х скрытых нейронов\n",w_hidden,"\n" )
    w_out = np.array([-1, 1, -0.5])
    print("вес для выходного нейрона\n",w_out,"\n" )

    su = np.dot(w_hidden, x)
    print("cкалярное произведение для скрытых нейронов")
    print(su,"\n")
    out = [act(x) for x in su]
    print("выходные значения скрытых")
    print(out,"\n")
    out.append(1)
    out = np.array(out)
    print("входные значения для выходного нейрона\n",out,"\n")

    su = np.dot(w_out, out)
    print("cкалярное для последнего нейронов")
    print(su,"\n")
    y = act(su)
    
    return y

C1 = [(1,0), (0,1)]
C2 = [(0,0), (1,1)]
print("выход\n",go(C1[0]))


#print( go(C1[0]), go(C1[1]) )
#print( go(C2[0]), go(C2[1]) )
