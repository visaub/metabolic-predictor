import os

import numpy as np
import pandas as pd
import copy

from explorer import Explorer
from models import write_traverse, add_energy, PL, ACSM, GG,  GG_running, PL_santee

from keras.layers import Dense, Dropout
from keras.models import load_model, Sequential

import matplotlib.pyplot as plt


input_names=['Weight','Load','Velocity','Slope']
# input_names=['Weight', 'Load', 'Velocity', 'Slope', 'Weight*Weight', 'Weight*Load', 'Load*Load', 'Weight*Velocity', 'Load*Velocity', 'Velocity*Velocity', 'Weight*Slope', 'Load*Slope', 'Velocity*Slope', 'Slope*Slope']   
normalization={'Weight':90.0, 'Load':50.0, 'Velocity':3.0, 'Slope':20.0, 'Rate':6000.0, 'HR':150.0}



def find_nn(E, input_names=['Weight','Load','Velocity','Slope'], epochs=100):

	model = Sequential()
	# Input - Layer
	model.add(Dense(200, activation='tanh', input_dim = (len(input_names))  ))   #activation='relu'
	
	# Hidden - Layers
	model.add(Dense(100, activation='tanh') )
	model.add(Dense(100, activation='tanh') )
	model.add(Dense(100, activation='tanh') )
	model.add(Dense(100, activation='tanh') )
	model.add(Dense(100, activation='tanh') )
	# model.add(layers.Dropout(0.05, noise_shape=None, seed=None))    #To avoid overfitting
	# model.add(Dense(50))#,activation='tanh'))
	# model.add(layers.Dropout(0.05, noise_shape=None, seed=None))    #To avoid overfitting
	model.add(Dense(10))
	# model.add(layers.Dropout(0.05, noise_shape=None, seed=None))    #To avoid overfitting
	

	# Output- Layer
	model.add(Dense(1, activation='sigmoid') ) #, activation = "relu"))


	model.summary()
	# compiling the model
	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])
	

	# Preparation of Inputs and Outputs
	inputs=E[input_names]

	# Normalization
	# normalization={'Weight':90.0, 'Load':50.0, 'Velocity':3.0, 'Slope':20.0, 'Rate':2000.0, 'HR':150.0}
	for k, feature in enumerate(input_names):
		for input_variable in feature.split('*'):
			inputs[:,k]/=normalization[input_variable]

	rate=E[['Rate']]
	rate/=normalization['Rate']     #  # rate/=2000.0     # Normalization

	results = model.fit(
		inputs, rate,
		epochs = epochs,
		# batch_size = 32,
		#validation_data = (inputs, rate),#(inputs_test, rate_test),
		verbose=1
	)

	model.save('trained_models/' + E.ID + '.h5')
	return model, results


def load_nn_model(name_model, normalization=normalization, input_names=input_names):
	class normalized_model():
		def __init__(self,model,normalization,input_names):
			self.model=model
			self.normalization = normalization
			self.input_names = input_names

		def predict(self,X):
			inputs_predict = copy.copy(X)
			# Normalization
			# normalization={'Weight':90.0, 'Load':50.0, 'Velocity':3.0, 'Slope':20.0, 'Rate':2000.0, 'HR':150.0}
			for k, feature in enumerate(self.input_names):
				for input_variable in feature.split('*'):
					inputs_predict[:,k]/=self.normalization[input_variable]

			y=self.model.predict(inputs_predict)
			y*=self.normalization['Rate']
			return y


	model = load_model('trained_models/'+name_model+'.h5' )
	return normalized_model(model, normalization, input_names)




	# # list all data in history
	# print(results.history.keys())
	# # summarize history for loss
	# plt.plot(results.history['loss'],'*-')
	# plt.plot(results.history['val_loss'],'*-')
	# plt.title('Model Loss over Time')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# score = model.evaluate(x_test, y_test, batch_size=128)
	# plt.show()
