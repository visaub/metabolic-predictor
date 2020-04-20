import flask
from flask import request, jsonify
import os
import pandas as pd
import numpy as np

from explorer import explorer

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return """<h1>Hello, the api is running</h1><p> Metabolic predictor api</p>
    <p>Author: Victor Sainz Ubide. Code: <a href='https://github.com/visaub/metabolic-predictor'>https://github.com/visaub/metabolic-predictor</a></p>
    """


@app.route('/api/subjects/', methods=['GET'])
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

	return jsonify([list_users_traverse,dict_users])


@app.route('/api/route/<ID>/<traverse>', methods=['GET'])
# @app.route('/api/route/<ID>/<Route>', methods=['GET'])
def route(ID=None, traverse=None): 
	# if 'id' in request.args:
	# 	ID = request.args['id']
	if not ID or not traverse:
		return "Error. Please, specify ID and traverse on the API, /api/route/<ID>/<traverse>"
	prefix='energy/temp/'
	if traverse+'.csv' not in os.listdir('energy/temp/'+ID):
		prefix='traverse/temp/'
		if traverse+'.csv' not in os.listdir('traverse/temp/'+ID):
			return "Error. Traverse does not exist"
		return "Error. Traverse has no energy"

	astronaut = explorer(problem_code=ID)
	df=astronaut.read_temp(ALL=False, traverse_name=traverse)[0]
	TIME=df['TIME']
	Weight=df['Weight']
	Load=df['Load']
	Velocity=df['Velocity']
	Slope=df['Slope']
	Eta=df['Eta']
	Rate=df['Rate']
	Fatigue=df['Fatigue']

	dict_return={'ID':ID, 'Traverse':traverse, 'elements':['TIME','Weight','Load','Velocity','Slope','Eta','Rate','Fatigue'] }
	dict_return['data']={}
	for i in range(len(TIME)):
		dict_return['data'][int(TIME[i])] = list(map(float, [Weight[i], Load[i], Velocity[i], Slope[i], Eta[i], Rate[i], Fatigue[i]] ))
	
	print(dict_return)

	# return jsonify({1:2,3:4})
	return jsonify(dict_return)



# Names until now: problem_code, id, ID, user,			please, change terminology



app.run(port=8800)