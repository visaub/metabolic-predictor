import numpy as np
import matplotlib.pyplot as plt
import sys

# from models import GG, PL, ACSM, SANTEE

list_graphs=['GG_grade plots', 'GG walking vs running', 'GG transition speed']


from models import GG as GG
from models import GG_running as GG_running
from models import PL as PL

# def GG(W,L,V=0.0, S=0.0, eta=1.0, g=9.8):
# 	# Givoni-Goldman model, 1971
# 	return eta*(W+L)*(2.3+0.32*max(V*3.6-2.5,0)**1.65+S*(0.2+0.07*max(V*3.6-2.5,0)))*4184/3600

# def GG_running(W,L,V=0.0, S=0.0, eta=1.0, g=9.8):
# 	mr = GG(W, L, V, S, eta=1.0, g=9.8)
# 	return (mr+0.47*(900*4184/3600 - mr))*(1+S/100) 




def massive_plots(title):

	# plt.subplots(figsize=(14, 7))
	# plt.subplot(131)

	if title=='GG_grade plots':
		Gs=np.arange(-12,25,4)
		Vs=np.linspace(2.5,9,100)
		W,L=70,0        #We don't care yet
		Ygg=np.zeros((len(Gs),len(Vs)))

		for ig,G in enumerate(Gs):
			for iv,V in enumerate(Vs):
				Ygg[ig,iv] = (GG(W,L,V/3.6,G,1.0)/(W+L)) * 3600/4184
			
			plt.plot( Vs, Ygg[ig, :], label=r'G = %d'%G )
			
		plt.title(title, fontsize=15)
		plt.xlabel(r'V $[km/h]$',fontsize=15)
		plt.ylabel(r'MR $kcal/(hr*kg)$', fontsize=15)
		plt.ylim(0,11) 
		plt.xlim(2.5,9) 
		plt.grid()
		plt.legend(fontsize=10)
		plt.show()

	if title=='GG walking vs running':

		W,L=70,0        #We don't care yet
		Vs=np.linspace(2.5,20,100)
		Y=np.zeros((2,len(Vs)))

		for iv,V in enumerate(Vs):
			Y[0,iv] = GG(W,L,V/3.6, 0, 1.0) * 3600/4184
			Y[1,iv] = GG_running(W,L,V/3.6, 0, 1.0) * 3600/4184
		
		plt.plot( Vs, Y[0, :], label = r'Walking' )
		plt.plot( Vs, Y[1, :], label = r'Running' )

		plt.title(title, fontsize=15)
		plt.xlabel(r'V $[km/h]$', fontsize=15)
		plt.ylabel(r'MR [kcal]', fontsize=15)
		# plt.ylim(0,11) 
		# plt.xlim(2.5,9) 
		plt.grid()
		plt.legend(fontsize=10)
		plt.show()


	if title == 'GG transition speed':

		def binary_search(funct, a, b, steps=20):
			if steps==0:
				return (a+b)/2
			if funct(a)>0:
				if funct((a+b)/2)>0:
					return binary_search(funct, (a+b)/2, b, steps-1)
				else:
					return binary_search(funct, a, (a+b)/2, steps-1)

			if funct(b)>0:
				if funct((a+b)/2)>0:
					return binary_search(funct, a, (a+b)/2, steps-1)
				else:
					return binary_search(funct, (a+b)/2, b, steps-1)

		plt.subplots(figsize=(14, 7))
		plt.subplot(121)

		W,L=70,0        #We don't care yet
		Gs=np.linspace(-20,20,100)   #Grades
		Y=np.zeros(len(Gs))

		for ig,G in enumerate(Gs):
			funct = lambda v: GG(W,L,v/3.6, G, 1.0) * 3600/4184 - GG_running(W,L,v/3.6, G, 1.0) * 3600/4184
			v=binary_search(funct, 0, 30, 20)
			Y[ig]=v
			# Y[ig] = GG(W,L,x/3.6, G, 1.0) * 3600/4184
		
		plt.plot(Gs, Y)

		plt.title(title, fontsize=15)
		plt.xlabel(r'Slope [%]', fontsize=15)
		plt.ylabel(r'Transition Speed [km/h]', fontsize=15)
		plt.grid()


		plt.subplot(122)

		W=50
		Ls=np.linspace(0,100,200)  #Loads
		Y=np.zeros(len(Gs))
		G=0

		for il, L in enumerate(Ls):
			funct = lambda v: GG(W,L,v/3.6, G, 1.0) * 3600/4184 - GG_running(W,L,v/3.6, G, 1.0) * 3600/4184
			v=binary_search(funct, 0, 30, 20)
			Y[il]=v

		plt.plot(Ls, Y)

		plt.title(title, fontsize=15)
		plt.xlabel(r'Weight+Load [kg]', fontsize=15)
		plt.ylabel(r'Transition Speed [km/h]', fontsize=15)
		plt.grid()
			

		plt.show()


if __name__ == '__main__':
	for i in range(len(list_graphs)):
		print (i,':',list_graphs[i])

	if len(sys.argv)==2:
		i = int(sys.argv[1])
	else:
		i = int(input('Select graph: enter Index from 0 to {}: '.format(len(list_graphs)) ))

	massive_plots(list_graphs[i])
