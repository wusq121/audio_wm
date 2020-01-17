import numpy as np
import Extraction as et

p5 = np.load('./batman-5min.npy')
p15 = np.load('./batman-15min.npy')
p30 = np.load('./batman-30min.npy')


P5 = et.pn_reconstruct(p5)/10
P15 = et.pn_reconstruct(p15)/10
P30 = et.pn_reconstruct(p30)/10

E5 = np.dot(P5, P5.T)
E15 = np.dot(P15, P15.T)
E30 = np.dot(P30, P30.T)

print(E5)
print(E15)
print(E30)