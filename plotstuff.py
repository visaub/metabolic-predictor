import numpy as np
import matplotlib.pyplot as plt
import sys


list_graphs=[
			'Individual Component',
			'GG_grade plots', 
			'PL_grade plots', 
			'GG walking vs running',
			'GG transition speed', 
			'PL_santee speed_grade plots',
			'Energy Expenditure (2019)',
			'Specfic Cost of Transport',
			'Speed Astronaut (Marquez, 2008)', 
			'SANTEE (2001)', 'SANTEE (2001) 2',
			'Cost of Transport, JMarquez+Santee',
			'Evolution of training',
			'Prediction model curves',
			'Train vs Test'
			]


from models import GG as GG
from models import GG_running as GG_running
from models import PL as PL
from models import PL_santee as PL_santee
from models import speed_astronaut as speed_astronaut
from models import SANTEE
from models import EE
from models import MODELS

from explorer import Explorer
from learn import find_nn, load_nn_model, normalization

def choose_and_plot(title):

	# This function contains different graphs of interest

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


	if title =='Specfic Cost of Transport':
		W,L=70,0        #We don't care yet
		Vs=np.linspace(0.01,5,1000)
		Y=np.zeros(len(Vs))

		G=0
		for model_key in ['PL', 'GG', 'PL_santee', 'EE']:
			model=MODELS[model_key]
			for iv,V in enumerate(Vs):
				Y[iv] = model(W=W, L=L, V=V, S=G, g=9.8)/(V*(W+L))
			
			plt.plot( Vs, Y, label=model_key)#+f', Optimal Speed=%.3f m/s'%Vs[list(Y).index(min(Y))] )
			# plt.plot( Vs[list(Y).index(min(Y))], min(Y), 'r*', label = f'Optimal Speed = %.3f m/s'%Vs[list(Y).index(min(Y))] )

		plt.title(r'Specfic Cost of Transport $[g=9.8m·s^{-2}]$', fontsize=15)
		plt.xlabel(r'V $[m/s]$', fontsize=15)
		plt.ylabel(r'Cost of Transport $[J·m^{-1}·kg^{-1}]$', fontsize=15)
		plt.xlim(0,6)
		plt.ylim(0,max(Y)*1.5)
		plt.grid()
		plt.legend(fontsize=10)
		plt.show()


	if title =='Energy Expenditure (2019)':
		W,L=70,0        #We don't care yet
		Gs=np.arange(24,-12.1, -4)
		Vs=np.linspace(0.1,8,100)
		Y=np.zeros(len(Vs))

		G=0
		# for ig, G in enumerate(Gs):
		for iv,V in enumerate(Vs):
			Y[iv] = EE(W=W, L=L, V=V, S=G, g=9.8)/(V*(W+L))
			
		plt.plot( Vs, Y, label=str(G)+'%')
		plt.plot( Vs[list(Y).index(min(Y))], min(Y), 'r*', label = f'Optimal Speed = %.3f m/s'%Vs[list(Y).index(min(Y))] )

		plt.title('Specific Cost of Transport using Energy Expenditure (2019)', fontsize=15)
		plt.xlabel(r'V $[m/s]$', fontsize=15)
		plt.ylabel(r'Cost of Transport $[J·m^{-1}·kg^{-1}]$', fontsize=15)
		plt.xlim(0,6)
		plt.ylim(0,max(Y)*1.5)
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
	   			Ys[iv,i]=SANTEE(80,0,V,S,eta=1.0,g=1.6)
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
	   			Ys[iv,i]=SANTEE(80,0,V,S,eta=1.0,g=1.6)
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


	if title=='Evolution of training':
		# CAREFUL
		# E=Explorer(ID='subject_test_11',ALL=False, traverse_name=['1','4','5','6','7','8','9','10','11'])
		E=Explorer(ID='user5')# , ALL=False, traverse_name=list(map(str, range(3,50))))
		E.recombine_features(4)


		input_names=['Weight','Load','Velocity','Slope']
		# input_names=['Load','Velocity','Slope']
		# input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']
		# TIME = E_test['TIME']



		# for epochs in [1,2,5,10,20,50,100,200,500,1000]:
		for epochs in [1000]:
			input_names=E.list_Xs[2:]
			_, results = find_nn(E, epochs=epochs*10, input_names=input_names)
			new_model = load_nn_model(E.ID, input_names=input_names)
			
			for i in ['1','4','6','7','8','10']:
				E_test=Explorer(ID='user5', ALL=False, traverse_name=i)
				E_test.recombine_features(4)
				
				input_names=E.list_Xs[2:]

				TIME = E_test['TIME']
				X = E_test[input_names]
				y = E_test['Rate']
				y_predicted = new_model.predict(X)
				print(X.shape)
				print(y_predicted.shape)
			
				plt.plot(TIME, y, 'b', label='Real, traverse: '+i)
				plt.plot(TIME, y_predicted, 'r', label='Estimation')

				plt.xlabel('TIME [s]')
				plt.ylabel('Metabolic Rate [W]')
				plt.grid()
				plt.title('Reality vs prediction, nºepochs='+str(epochs))
				plt.xlim(0,TIME[-1])
				plt.ylim(0.6*min(min(y_predicted),min(y)), 1.25*max(max(y_predicted),max(y)))

				plt.legend(fontsize=10)
			
				plt.show()



	if title=='Prediction model curves':
		input_names=['Weight','Load','Velocity','Slope']
		E=Explorer(ID='user5')
		E.recombine_features(4)
		input_names=E.list_Xs[2:]
		# E=Explorer(ID='subject_test_13')
		# find_nn(E, epochs=100, input_names=input_names)
		# find_nn(E, input_names=input_names)
		new_model = load_nn_model(E.ID, input_names=input_names)

		Gs=np.arange(12,-8.1, -4)

		samples = 1000
		Vs=np.linspace(0, 5, samples)  
		W,L = float(E.weight), 0.0        

		print(len(new_model.input_names))
		for S in Gs:
			X={'Weight':W, 'Load':L, 'Velocity':Vs, 'Slope':S, 'length':samples}
			Y = new_model.predict(X)
			print(Y.shape)
			print(Vs.shape)

			plt.plot( Vs, Y[:,0]/Vs, label='Slope = {}%'.format(S) )

		# Y = new_model.predict(X)

			
		plt.title('MCT: Prediction', fontsize=15)
		plt.xlabel(r'V $[m/s]$',fontsize=15)
		plt.ylabel(r'MCT [W/m]', fontsize=15)
		# plt.ylim(0,12) 
		plt.xlim(0,4) 
		plt.grid()
		plt.legend(fontsize=10)  #change pos soon
		plt.show()


	if title == 'Train vs Test':
		input_names=['Weight', 'Load', 'Velocity', 'Slope']
		#input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']		
		E = Explorer(ID='user5', input_model='PL_santee', num_iters=30, ALL=False, traverse_name=list(map(str,range(11,32))) )
		E_test = Explorer(ID='user5', ALL=False, traverse_name=['0','1','2','3','4','5','6','7','8','9','10'])

		n=4
		E_test.recombine_features(n)
		E.recombine_features(n)

		input_names=E.list_Xs[2:]
		print(input_names)

		# E=Explorer(ID='subject_test_13')
		# find_nn(E, epochs=100, input_names=input_names)
		model, results = find_nn(E, input_names=input_names, E_test=E_test, layout=[200,50,1], epochs=1000)
		new_model = load_nn_model(E.ID, input_names=input_names)

		print(results.history.keys())   # Debugging

		plt.plot(np.sqrt(np.array(results.history['loss']))*normalization['Rate'],'*-',label='train')
		plt.plot(np.sqrt(np.array(results.history['val_loss']))*normalization['Rate'],'*-',label='test')
		plt.xlabel('Iterations')
		plt.ylabel('Root Mean Squared Error [W]')
		# plt.legend(['train', 'test'], loc='upper right')
		plt.legend(loc='upper right')
		plt.title('Learning: train-set vs test-set, n = '+str(n))
		plt.xlim(0,)
		plt.ylim(0,)
		plt.grid()
		plt.show()


		samples = 1000
		Vs=np.linspace(0, 5, samples)  
		W,L = float(E.weight), 0.0        
		S = 0.0
		# X=np.zeros((len(Vs),len(input_names)))
		X={'Weight':W, 'Load':L, 'Velocity':Vs, 'Slope':S}
		X['length'] = samples

		Y_0 = np.zeros(len(Vs))
		
		for iv,V in enumerate(Vs):
			# X[iv,:] = np.array([W,L,V,S])
			Y_0[iv] = PL_santee(W,L,V,S)


		# input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']

		Y = new_model.predict(X)
		print(new_model.input_names)

		plt.plot( Vs, Y_0/Vs, 'b', label='Real' )
		plt.plot( Vs, Y[:,0]/Vs, 'r', label='Prediction' )


		# Y = new_model.predict(X)

			
		plt.title('MCT', fontsize=15)
		plt.xlabel(r'V $[m/s]$',fontsize=15)
		plt.ylabel(r'MCT [W/m]', fontsize=15)
		# plt.ylim(0,12) 
		plt.xlim(0,4) 
		plt.grid()
		plt.legend(fontsize=10)  #change pos soon
		plt.show()



	if title=='Individual Component':
		available=['Weight','Load','Velocity','Slope','Rate']
		print(available)
		
		# t = input('Please, choose traverse: ')
		# E = Explorer(ID='user5', input_model='PL_santee', num_iters=30, ALL=False, traverse_name=t, weight=70)
		E = Explorer(ID='user5', input_model='PL_santee', num_iters=30, ALL=False, traverse_name=['1','2','3','4','5'], weight=70)

		for i in range(len(available)):			
			# plt.title('User1, traverse '+str(t), fontsize=15)
			plt.title('User1')
			for t in range(20,26):
				E = Explorer(ID='user5', input_model='PL_santee', num_iters=30, ALL=False, traverse_name=str(t), weight=70)
				plt.plot(E['TIME'], E[available[i]], label='traverse ' +str(t))
			plt.xlabel('TIME [seconds]', fontsize=15)
			plt.ylabel(available[i], fontsize=15)
			plt.legend(fontsize=10)
			if available[i]!='Slope':
				plt.ylim(0,) 
			else:
				plt.ylim(-20,20)
			plt.xlim(0,7200) 
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
