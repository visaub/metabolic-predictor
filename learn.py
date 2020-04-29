import os

import numpy as np
import pandas as pd
import copy

from explorer import explorer
from models import write_traverse, add_energy, PL, ACSM, GG,  GG_running, PL_santee

from keras import layers
from keras.models import load_model, Sequential

import matplotlib.pyplot as plt


input_names=['Weight','Load','Velocity','Slope']
# input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']



def find_nn(E, input_names=['Weight','Load','Velocity','Slope'], epochs=100):

	model = Sequential()
	# Input - Layer
	model.add(layers.Dense(200, activation = 'relu', input_dim = (len(input_names) )))
	
	# Hidden - Layers
	# model.add(layers.Dropout(0.01, seed=None))
	# model.add(layers.Dense(1000, activation = "sigmoid", kernel_initializer='normal'))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))   #
	model.add(layers.Dense(100))
	model.add(layers.Dense(50))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	# model.add(layers.Dense(100, activation = "relu"))
	# model.add(layers.Dropout(0.01, noise_shape=None, seed=None))
	# model.add(layers.Dense(100, activation = "sigmoid"))

	# Output- Layer
	model.add(layers.Dense(1))  #, activation = "relu"))


	model.summary()
	# compiling the model
	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])
	

	# Preparation of Inputs and Outputs
	inputs=E[input_names]

	# Normalization
	inputs[:,0]/=90.0
	inputs[:,1]/=50.0
	inputs[:,2]/=2.0
	inputs[:,3]/=10.0

	rate=E[['Rate']]
	rate/=500.0     # Normalization

	results = model.fit(
		inputs, rate,
		epochs = epochs,
		# batch_size = 32,
		#validation_data = (inputs, rate),#(inputs_test, rate_test),
		verbose=1
	)

	model.save('trained_models/' + E.ID + '.h5')
	return model, results


def load_nn_model(name_model):
	class normalized_model():
		def __init__(self,model):
			self.model=model

		def predict(self,X):
			X_normalized = copy.copy(X)
			X_normalized[:,0] = X[:,0]/90.0
			X_normalized[:,1] = X[:,1]/50.0
			X_normalized[:,2] = X[:,2]/2.0
			X_normalized[:,3] = X[:,3]/10.0
			y=self.model.predict(X_normalized)
			y*=500.0
			return y

	model = load_model('trained_models/'+name_model+'.h5' )
	return normalized_model(model)




	# # list all data in history
	# print(results.history.keys())
	# # summarize history for loss
	# plt.plot(results.history['loss'],'*-')
	# plt.plot(results.history['val_loss'],'*-')
	# plt.title('Model Loss over Time')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')

	# plt.show()


	# # Create input variables matrix
	# # ['Weight','Load','Velocity','Slope']

	# L=50000
	# Weight=np.linspace(80,80,L)
	# Load=np.linspace(0,0,L)
	# Velocity=np.linspace(2,2,L)
	# Slope=np.linspace(-20,20,L)

	# d_inputs = {'Weight':Weight, 'Load':Load, 'Velocity':Velocity, 'Slope':Slope}

	# inputs_matrix=np.zeros((L,14))
	# inputs_matrix[:,0]=Weight
	# inputs_matrix[:,1]=Load
	# inputs_matrix[:,2]=Velocity
	# inputs_matrix[:,3]=Slope
	# for i in range(0,len(input_names)):
	# 	column = d_inputs[input_names[i].split('*')[0]]
	# 	for j in range(len(input_names[i].split('*'))-1):
	# 		column *= d_inputs[input_names[i].split('*')[j]]
	# 	inputs_matrix[:,i] = column

	# input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']

	# Predicted_Rate=model.predict(x=inputs_matrix)[:,0]

	# #PL(W,L=0.0, V=0.0, G=0.0, eta=1.0)
	# Rate=np.array(list(map(lambda e: PL_santee(*e),inputs_matrix[:,0:4])))

	# np.concatenate((Rate,Predicted_Rate))
	# ylim=(min(np.concatenate((Rate,Predicted_Rate))),max(np.concatenate((Rate,Predicted_Rate))))


	# fig, ax = plt.subplots(figsize=(12, 7))
	# ax.plot(Slope,Predicted_Rate,'b',label='Neural Net')
	# ax.plot(Slope,Rate,'r',label='Original Model')
	# ax.set_xlabel('Slope [degrees]')
	# ax.set_ylabel('Power [W]')
	# plt.ylim(ylim)
	# ax.set_title('Slope')
	# fig.legend(fontsize=15)
	# ax.grid()


	# Velocity=np.linspace(0,5,L)
	# Slope=np.linspace(0,0,L)
	# inputs_matrix[:,2]=Velocity
	# inputs_matrix[:,3]=Slope

	# Predicted_Rate=model.predict(x=inputs_matrix)[:,0]
	# Rate=np.array(list(map(lambda e: PL_santee(*e),inputs_matrix[:,0:4])))

	# ylim=(min(np.concatenate((Rate,Predicted_Rate))),max(np.concatenate((Rate,Predicted_Rate))))

	# fig, ax = plt.subplots(figsize=(12, 7))
	# ax.plot(Velocity,Predicted_Rate,'b',label='Neural Net')
	# ax.plot(Velocity,Rate,'r',label='Original Model')
	# ax.set_xlabel('Velocity [m/s]')
	# ax.set_ylabel('Power [W]')
	# ax.set_title('Velocity')
	# plt.ylim(ylim)
	# fig.legend(fontsize=15)
	# ax.grid()
	# plt.show()