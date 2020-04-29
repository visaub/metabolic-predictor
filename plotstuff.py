import numpy as np
import matplotlib.pyplot as plt
import sys

# from models import GG, PL, ACSM, SANTEE

list_graphs=['GG_grade plots', 'PL_grade plots', 'GG walking vs running', 'GG transition speed', 'PL_santee speed_grade plots','Speed Astronaut (Marquez, 2008)', 'SANTEE (2001)', 'SANTEE (2001) 2', 'Cost of Transport, JMarquez+Santee']


from models import GG as GG
from models import GG_running as GG_running
from models import PL as PL
from models import PL_santee as PL_santee
from models import speed_astronaut as speed_astronaut
from models import SANTEE

def choose_and_plot(title):

	# plt.subplots(figsize=(14, 7))
	# plt.subplot(131)

	if title=='Speed Astronaut (Marquez, 2008)':
		Ss=np.linspace(-25,25,200)     #Slopes
		Vs=np.zeros(len(Ss))
		for i, S in enumerate(Ss):
			Vs[i] = speed_astronaut(S)

		plt.plot(Ss,Vs)
		plt.title(title, fontsize=15)
		plt.xlabel('Slope [%]')
		plt.ylabel('Speed [m/s]')
		plt.grid()
		plt.show()

	if title=='GG_grade plots':
		Gs=np.arange(-12,25,4)
		Vs=np.linspace(2.5,9,100)     #km/h
		W,L=70,0        #We don't care yet
		Ygg=np.zeros((len(Gs),len(Vs)))

		for ig,G in enumerate(Gs):
			for iv,V in enumerate(Vs):
				Ygg[ig,iv] = (GG(W,L,V/3.6,G,1.0)/(W+L)) * 3600/4184
			
			plt.plot( Vs, Ygg[ig, :], label='G = {}%'.format(G) )
			
		plt.title(title, fontsize=15)
		plt.xlabel(r'V $[km/h]$',fontsize=15)
		plt.ylabel(r'MR $kcal/(hr*kg)$', fontsize=15)
		plt.ylim(0,11) 
		plt.xlim(2.5,9) 
		plt.grid()
		plt.legend(fontsize=10)
		plt.show()


	if title=='PL_grade plots':
		Gs=np.arange(24,-4.1, -4)
		Vs=np.linspace(0, 20, 100)   #km/h
		W,L=70,0        #We don't care yet
		Ygg=np.zeros((len(Gs),len(Vs)))

		for ig,G in enumerate(Gs):
			for iv,V in enumerate(Vs):
				Ygg[ig,iv] = (PL(W,L,V/3.6,G,1.0)/(W+L)) * 3600/4184
			
			plt.plot( Vs, Ygg[ig, :], label='G = {}%'.format(G) )
			
		plt.title(title, fontsize=15)
		plt.xlabel(r'V $[km/h]$',fontsize=15)
		plt.ylabel(r'MR $kcal/(hr*kg)$', fontsize=15)
		plt.ylim(0,12) 
		plt.xlim(2,12) 
		plt.grid()
		plt.legend(fontsize=10)  #change pos soon
		plt.show()
	

	if title=='PL_santee speed_grade plots':

		Vs=np.arange(0,10,1)   #km/h
		Gs=np.linspace(-20, 20, 5000)
		W,L=70,0        #We don't care yet
		Ygg=np.zeros((len(Vs),len(Gs)))

		for iv,V in enumerate(Vs):
			for ig,G in enumerate(Gs):
				Ygg[iv,ig] = (PL_santee(W,L, V/3.6, G, 1.0)/(W+L)) * 3600/4184
			
			plt.plot( Gs, Ygg[iv, :], label='V = {}km/h'.format(V) )
		
		plt.title(title+' W={}, L={}'.format(W,L) , fontsize=15)
		plt.xlabel(r'G $[%]$',fontsize=15)
		plt.ylabel(r'MR $kcal/(hr*kg)$', fontsize=15)
		#plt.ylim(0,14) 
		#plt.xlim(0,12) 
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
		Ls=np.linspace(50,100,200)  #Loads
		Y=np.zeros(len(Ls))
		G=0

		for il, L in enumerate(Ls):
			funct = lambda v: GG(W,L,v/3.6, G, 1.0) * 3600/4184 - GG_running(W,L,v/3.6, G, 1.0) * 3600/4184
			v=binary_search(funct, 0, 30, 20)
			Y[il]=v

		plt.plot(Ls, Y, 'r')

		plt.title(title, fontsize=15)
		plt.xlabel(r'Weight+Load [kg]', fontsize=15)
		plt.ylabel(r'Transition Speed [km/h]', fontsize=15)
		plt.grid()
			

		plt.show()


	if title=='SANTEE (2001)':
		Ss=np.arange(-20,20,0.1)
		Vs=np.linspace(0,5,6)
		Ys=np.zeros((len(Vs),len(Ss)))
		for iv,V in enumerate(Vs):
			for i,S in enumerate(Ss):
	   			Ys[iv,i]=SANTEE(80,0,V,S,1.0,g=1.6)
			plt.plot(Ss, Ys[iv, :], label=r'V = '+str(V)+' m/s')
		plt.title(title, fontsize=15)
		plt.xlabel(r'Slope [%]', fontsize=15)	
		plt.ylabel(r'Metabolic Rate [W]', fontsize=15)
		plt.grid()
		plt.legend(fontsize=10)
		plt.show()

	if title=='SANTEE (2001) 2':
		Ss=np.arange(-20,20,4)
		Vs=np.linspace(0,5,100)
		Ys=np.zeros((len(Vs),len(Ss)))
		for i,S in enumerate(Ss):
			for iv,V in enumerate(Vs):
	   			Ys[iv,i]=SANTEE(80,0,V,S,1.0,g=1.6)
			plt.plot(Vs, Ys[:, i], label=r'S = '+str(S)+'%')
		plt.title(title, fontsize=15)
		plt.xlabel(r'Velocity [m/s]', fontsize=15)	
		plt.ylabel(r'Metabolic Rate [W]', fontsize=15)
		plt.grid()
		plt.legend(fontsize=10)
		plt.show()

	if title=='Cost of Transport, JMarquez+Santee':
		Ss=np.arange(-30,30,0.1)
		# Ss=[-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
		Cost=np.zeros(len(Ss))
		for i,S in enumerate(Ss):
			V=speed_astronaut(S)
			Cost[i] = SANTEE(80,20,V,S,g=1.6)/V
		plt.plot(Ss,Cost)
		plt.title(title, fontsize=15)
		plt.xlabel(r'Slope [%]', fontsize=15)	
		plt.ylabel(r'Energy [J/m]]', fontsize=15)
		plt.grid()
		plt.show()


if __name__ == '__main__':
	for i in range(len(list_graphs)):
		print (i,':',list_graphs[i])

	if len(sys.argv)==2:
		i = int(sys.argv[1])
	else:
		i = int(input('Select graph: enter Index from 0 to {}: '.format(len(list_graphs)-1) ))

	choose_and_plot(list_graphs[i])
