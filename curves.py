

import numpy as np
import matplotlib.pyplot as plt

from models import GG, PL, ACSM, SANTEE

Gs=np.arange(-12,25,4)
Vs=np.linspace(0,2.4,1000)
W,L=70,0        #We don't care yet

Ypl=np.zeros((len(Gs),len(Vs)))

plt.subplots(figsize=(14, 7))
plt.subplot(131)


for ig,G in enumerate(Gs):
    for iv,V in enumerate(Vs):
        Ypl[ig,iv]=PL(W,L,V,G,1.0)
    plt.plot(Vs, Ypl[ig, :],label=r'G = %d'%G)

plt.title('Pandolf Equation',fontsize=15)
plt.xlabel(r'V $[m/s]$',fontsize=15)
plt.ylabel(r'MR $[Watt]$',fontsize=15)
plt.xlim(0,2.4) 
plt.ylim(0,1100) 
plt.grid()
plt.legend(fontsize=10)

Vs=np.linspace(2.5,9,1000)/3.6

Ygg=np.zeros((len(Gs),len(Vs)))

# fig, ax = plt.subplots(figsize=(7, 7))
plt.subplot(132)
for ig,G in enumerate(Gs):
    for iv,V in enumerate(Vs):
        Ygg[ig,iv]=(GG(W,L,V,G,1.0)/(W+L))*3600/4184
    
    plt.plot(Vs*3.6, Ygg[ig, :],label=r'G = %d'%G)

plt.title('Givoni Goldman',fontsize=15)
plt.xlabel(r'V $[km/h]$',fontsize=15)


plt.ylabel(r'MR $kcal/(hr*kg)$', fontsize=15)
plt.ylim(0,11) 
plt.grid()
plt.legend(fontsize=10)

# plt.show()

# fig, ax = plt.subplots(figsize=(7, 7))
plt.subplot(133)

Gs=np.arange(-30,30,0.1)
Vs=np.linspace(0,10,5)
Ys=np.zeros((len(Vs),len(Gs)))
for iv,V in enumerate(Vs):
	for ig,G in enumerate(Gs):
	   Ys[iv,ig]=SANTEE(W,L,V,G,1.0,g=1.6)
	plt.plot(Gs, Ys[iv, :], label=r'V = '+str(V))

plt.title('SANTEE',fontsize=15)
plt.xlabel(r'Slope %',fontsize=15)


plt.ylabel(r'MR $[W]$', fontsize=15)
# plt.ylim(0,11) 
plt.grid()
plt.legend(fontsize=10)
plt.show()

plt.subplots_adjust(top=1.5, bottom=0, left=0.10, right=1.7, hspace=0.425,wspace=0.35)


