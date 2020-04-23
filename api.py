import flask
from flask import request, jsonify
import os
import pandas as pd
import numpy as np
import requests as r

from keras import layers       # We have to use Theano instead of Tensorflow
from keras.models import load_model, Sequential

from explorer import explorer
from learn import find_nn

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return """<h1>Hello, the api is running</h1><p> Metabolic predictor api</p>
    <p>Author: Victor Sainz Ubide. Code: <a href='https://github.com/visaub/metabolic-predictor'>https://github.com/visaub/metabolic-predictor</a></p>
    """

@app.route('/api/endpoints')
def docs():
	return """
	'/api/subjects?id=id&type=list'
	<br>
	'/api/route/ID/traverse'
	<br>
	'/api/route/', methods=['POST']
	<br>
	'/api/predict/', methods=['POST']
	"""

@app.route('/api/subjects', methods=['GET'])
def subjects(kind=''): 
	if 'type' in request.args and request.args['type']=='list':
		list_users_traverse = os.listdir('traverse/temp')
		return jsonify(list_users_traverse)
	ID = None
	if 'id' in request.args:
		ID = request.args['id']
	list_users_traverse = os.listdir('traverse/temp')
	list_users_energy = os.listdir('energy/temp')
	dict_users= {}
	if ID:
		if ID in list_users_traverse:
			list_users_traverse=[ID]
		else: 
			return "Error. There is no subject with an id: <b>"+ID+"</b>"
	for u in list_users_traverse:
		if u not in dict_users:
			dict_users[u] = {}
		for traverse in os.listdir('traverse/temp/'+u):
			if traverse in os.listdir('energy/temp/'+u):
				dict_users[u][traverse]=True
			else:
				dict_users[u][traverse]=False

	return jsonify({'subjects':list_users_traverse,'data':dict_users})


@app.route('/api/route/<ID>/<traverse>', methods=['GET'])
@app.route('/api/route/<ID>', methods=['GET'])
def get_route(ID=None, traverse=None): 
	# if 'id' in request.args:
	# 	ID = request.args['id']
	if not ID or not traverse:
		return "Error. Please, specify ID and traverse on the API, <b>/api/route/ID/traverse</b>"
	prefix='energy/temp/'
	if ID not in os.listdir('traverse/temp/'):
		return "Error. Subject with ID: <b>"+ID+"</b> is not registered"
	if traverse+'.csv' not in os.listdir('energy/temp/'+ID):
		prefix='traverse/temp/'
		if traverse+'.csv' not in os.listdir('traverse/temp/'+ID):
			return "Error. Traverse does not exist"
		return "Error. Traverse has no energy"

	astronaut = explorer(ID=ID)
	df=astronaut.read_temp(ALL=False, traverse_name=traverse)[0]
	TIME=df['TIME']
	weight=df['Weight']
	load=df['Load']
	velocity=df['Velocity']
	slope=df['Slope']
	eta=df['Eta']
	rate=df['Rate']
	fatigue=df['Fatigue']

	dict_return={'ID':ID, 'Traverse':traverse, 'elements':['TIME','Weight','Load','Velocity','Slope','Eta','Rate','Fatigue'] }
	dict_return['traverse type'] = 'energy'    # Not final yet
	
	dict_return['data']={}
	for i in range(len(TIME)):
		dict_return['data'][int(TIME[i])] = {'Weight':float(weight[i]), 
											'Load':float(load[i]), 
											'Velocity':float(velocity[i]), 
											'Slope':float(slope[i]), 
											'Eta':float(eta[i]), 
											'Rate':float(rate[i]), 
											'Fatigue':float(fatigue[i])
											}

	return jsonify(dict_return)



@app.route('/api/route', methods=['POST'])
# Add traverse to the database
def add_route():
	if request.method == 'POST':
		list_users_traverse = os.listdir('traverse/temp')
		json = request.json
		if "ID" not in json or "elements" not in json or "data" not in json or 'traverse type' not in json:
			return "Error. Submission incomplete"
		ID = json["ID"]
		elements=json["elements"]
		traverse_type=json["traverse type"]

		prefix='temp/'+ID+'/'
		if ID not in list_users_traverse:
			os.mkdir('traverse/'+prefix)
			os.mkdir('energy/'+prefix)
			print("Created new subject: ID = "+ID)

		if "Traverse" in json:
			filename = json["Traverse"]
		else:
			filename = str(len(os.listdir('traverse/'+prefix) ) +1)


		data=json["data"]
		TIME=list(map(int,data.keys()))
		TIME.sort()
		TIME=list(map(str,TIME))
		weight,load,velocity,slope,eta,gravity=[],[],[],[],[],[]
		x,y=[],[]

		for t in TIME:
			x.append(0)    # Points should be deprecated
			y.append(0)
			weight.append(data[t]['Weight'])
			load.append(data[t]['Load'])
			velocity.append(data[t]['Velocity'])
			slope.append(data[t]['Slope'])
			eta.append(0.7)
			gravity.append(9.6)

		df = pd.DataFrame({	'TIME': TIME,
			'X':x,
			'Y':y,
			'Weight': weight,
			'Load': load,
			'Velocity': velocity,
			'Slope': slope,
			'Eta': eta,
			'Gravity': gravity})

		df.to_csv('traverse/' + prefix + filename + '.csv', index=False, columns=['TIME','X','Y','Weight','Load','Velocity','Slope','Eta','Gravity'])
		# df.to_csv('energy/'+ prefix + filename + '.csv', index=False, columns=['TIME','Weight','Load','Velocity','Slope','Eta','Gravity','Rate','Fatigue'])
		
		# write_traverse(l_points, filename='', velocity=1.25, weight=80, load=0, precision=60, eta=1.0, gravity=9.8)


		# traverse =json["Traverse"]
		# data = json["data"]

		# write to energy

	return ("OK. Traverse "+filename+" added to subject: "+ID,200)


@app.route('/api/predict', methods=['POST'])
#Predict Rate at an instance
def predict():

	json = request.json
	if "ID" not in json or "data" not in json:
		return "Error. Submission incomplete"
	ID = json["ID"]
	data=json["data"]
	if ID not in os.listdir('traverse/temp/'):
		return "Error. Subject with ID: <b>"+ID+"</b> is not registered"
	
	TIME=list(map(int,data.keys()))
	TIME.sort()
	TIME=list(map(str,TIME))
	weight,load,velocity,slope,eta,gravity=[],[],[],[],[],[]

	for t in TIME:
		weight.append(data[t]['Weight'])
		load.append(data[t]['Load'])
		velocity.append(data[t]['Velocity'])
		slope.append(data[t]['Slope'])
		# eta.append(data[t]['Eta'])

	E=explorer(ID = ID)
	input_names = ['1', 'Weight', 'Load', 'Velocity', 'Slope']
	
	if E.ID+'.h5' not in os.listdir('trained_models/'):
		model, results = find_nn(E, input_names, epochs=10)
	else:
		model = load_model('trained_models/'+E.ID+'.h5')

	x = E[input_names]
	y = model.predict(x)

	dict_return = json
	dict_return['Rate Predicted']=[]
	for i in range(len(TIME)):
		dict_return['data'][TIME[i]]['Rate Predicted'] = float(y[i])
		dict_return['Rate Predicted'].append(float(y[i]))
	return jsonify(dict_return)


	# df = pd.DataFrame({	'TIME': TIME,
	# 	'X':x,
	# 	'Y':y,
	# 	'Weight': weight,
	# 	'Load': load,
	# 	'Velocity': velocity,
	# 	'Slope': slope,
	# 	'Eta': eta,
	# 	'Gravity': gravity})

@app.route('/test')
def testtt():
	model = load_model('trained_models/GG.h5')	
	return 'GG'



if __name__ == '__main__':
	app.run(port=8800)