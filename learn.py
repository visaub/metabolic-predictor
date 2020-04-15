import os

import numpy as np
import pandas as pd

from explorer import explorer
from models import write_traverse, add_energy, PL, ACSM, GG,  GG_running, PL_santee

from keras import layers
from keras.models import load_model, Sequential

import matplotlib.pyplot as plt


#input_names=['Weight','Load','Velocity','Slope']
input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']



def find_nn(E, input_names, name_model=None):

	model = Sequential()

	# Input - Layer
	model.add(layers.Dense(100, activation = "sigmoid", input_shape=(len(input_names), )))
	# Hidden - Layers
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	model.add(layers.Dense(100, activation = "sigmoid"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	model.add(layers.Dense(100, activation = "sigmoid"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	model.add(layers.Dense(100, activation = "sigmoid"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	# model.add(layers.Dense(200, activation = "sigmoid"))
	# Output- Layer
	model.add(layers.Dense(1, activation = "relu"))
	model.summary()
	# compiling the model
	model.compile(
		optimizer = 'adam',
		loss = "mean_squared_error",
		metrics = ["mean_squared_error"]
	)

	inputs=E[input_names]
	rate=E[['Rate','Fatigue']]

	results = model.fit(
		inputs, rate,
		epochs= 20,
		# batch_size = 32,
		validation_data = (inputs, rate),#(inputs_test, rate_test),
		verbose=1
	)
	if name_model:
		model.save('trained_models/'+name_model+'.h5')
	return results




if __name__ == '__main__':

	E1=explorer(input_model='PL_santee', num_iters=50, problem_code='train_PL')
	E1.recombine_features(2)
	inputs=E1[input_names]
	rate=E1['Rate']

	E2=explorer(input_model='PL_santee', num_iters=100, problem_code='test_PL')
	E2.recombine_features(2)
	inputs_test=E2[input_names]
	rate_test=E2['Rate']

	print('Creating model...')

	model = Sequential()

	# Input - Layer
	model.add(layers.Dense(100, activation = "sigmoid", input_shape=(len(input_names), )))
	# Hidden - Layers
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	model.add(layers.Dense(100, activation = "sigmoid"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	model.add(layers.Dense(100, activation = "sigmoid"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	model.add(layers.Dense(100, activation = "sigmoid"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	# model.add(layers.Dense(250, activation = "sigmoid"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	# model.add(layers.Dense(200, activation = "sigmoid"))
	# model.add(layers.Dense(300, activation = "relu"))
	# model.add(layers.Dense(20, activation = "sigmoid"))

	# Output- Layer
	model.add(layers.Dense(1, activation = "relu"))
	model.summary()
	# compiling the model

	model.compile(
		optimizer = 'adam',
		loss = "mean_squared_error",
		metrics = ["mean_squared_error"]
	)

	results = model.fit(
		inputs, rate,
		epochs= 20,
		# batch_size = 32,
		validation_data = (inputs_test, rate_test),
		verbose=1
	)



	aux=input('Press Enter key to continue...')


	### SAVING/LOADING MODEL ###

	name_model='PL_santee'
	model.save('trained_models/'+name_model+'.h5')
	print('Deleting model...')
	del model

	print('Loading model again...')
	model=load_model('trained_models/'+name_model+'.h5')
		

	# list all data in history
	print(results.history.keys())
	# summarize history for loss
	plt.plot(results.history['loss'],'*-')
	plt.plot(results.history['val_loss'],'*-')
	plt.title('Model Loss over Time')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')

	plt.show()


	# Create input variables matrix
	# ['Weight','Load','Velocity','Slope']

	L=50000
	Weight=np.linspace(80,80,L)
	Load=np.linspace(0,0,L)
	Velocity=np.linspace(2,2,L)
	Slope=np.linspace(-20,20,L)

	d_inputs = {'Weight':Weight, 'Load':Load, 'Velocity':Velocity, 'Slope':Slope}

	inputs_matrix=np.zeros((L,14))
	inputs_matrix[:,0]=Weight
	inputs_matrix[:,1]=Load
	inputs_matrix[:,2]=Velocity
	inputs_matrix[:,3]=Slope
	for i in range(0,len(input_names)):
		column = d_inputs[input_names[i].split('*')[0]]
		for j in range(len(input_names[i].split('*'))-1):
			column *= d_inputs[input_names[i].split('*')[j]]
		inputs_matrix[:,i] = column

	input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']

	Predicted_Rate=model.predict(x=inputs_matrix)[:,0]

	#PL(W,L=0.0, V=0.0, G=0.0, eta=1.0)
	Rate=np.array(list(map(lambda e: PL_santee(*e),inputs_matrix[:,0:4])))

	np.concatenate((Rate,Predicted_Rate))
	ylim=(min(np.concatenate((Rate,Predicted_Rate))),max(np.concatenate((Rate,Predicted_Rate))))


	fig, ax = plt.subplots(figsize=(12, 7))
	ax.plot(Slope,Predicted_Rate,'b',label='Neural Net')
	ax.plot(Slope,Rate,'r',label='Original Model')
	ax.set_xlabel('Slope [degrees]')
	ax.set_ylabel('Power [W]')
	plt.ylim(ylim)
	ax.set_title('Slope')
	fig.legend(fontsize=15)
	ax.grid()


	Velocity=np.linspace(0,5,L)
	Slope=np.linspace(0,0,L)
	inputs_matrix[:,2]=Velocity
	inputs_matrix[:,3]=Slope

	Predicted_Rate=model.predict(x=inputs_matrix)[:,0]
	Rate=np.array(list(map(lambda e: PL_santee(*e),inputs_matrix[:,0:4])))

	ylim=(min(np.concatenate((Rate,Predicted_Rate))),max(np.concatenate((Rate,Predicted_Rate))))

	fig, ax = plt.subplots(figsize=(12, 7))
	ax.plot(Velocity,Predicted_Rate,'b',label='Neural Net')
	ax.plot(Velocity,Rate,'r',label='Original Model')
	ax.set_xlabel('Velocity [m/s]')
	ax.set_ylabel('Power [W]')
	ax.set_title('Velocity')
	plt.ylim(ylim)
	fig.legend(fontsize=15)
	ax.grid()
	plt.show()