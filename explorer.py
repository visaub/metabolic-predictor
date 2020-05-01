
from models import write_traverse, add_energy, PL, GG,  GG_running, PL_santee
import os
import numpy as np
import pandas as pd

# At the moment I'll be using Pandolf Load model with L=0 and eta=1.0 for the examples
# PL(W, L=0.0, V=0.0, S=0.0, eta=1.0) (becomes)-> 1.5*W + (W+L)*(1.5*V^2+0.35*V*S)


class Environment():
	"""environment, specially important are gravity and eta"""
	def __init__(self, gravity=9.8,eta=1.0,weight=None,load=None):
		self.gravity = gravity
		self.eta=eta
		self.weight=weight
		self.load=load


class Explorer():    # Most important Class
	def __init__(self,input_model='PL',num_iters=10, ID=False, gravity=9.8, weight=None, load=None, eta=1.0):
		env=Environment(gravity,eta,weight,load)
		if ID:
			self.ID = ID
			list_users_traverse = os.listdir('traverse/temp')
			if ID not in list_users_traverse:
				try:
					self.ID = generate_traverses(num_iters, ID=ID, input_model=input_model, env=env) 
					print("Created new subject: ID = "+ID)
				except FileExistsError:
					pass

		if not ID:
			self.ID = generate_traverses(num_iters, input_model=input_model, env=env)

		self.list_dfs=self.read_temp()
		self.list_Xs=['TIME','Weight','Load','Velocity','Slope']
		self.list_Ys=['Rate']
		#my_dataframe.columns.values
		self.df=pd.concat(self.list_dfs)
		self.Y=np.array(self.df['Rate'])
		self.X=np.array(self.df[self.list_Xs])
		self.recombined=1
		self.recombine_features(n=1)
		self.env=env

	def __getitem__(self,column):
		return np.array(self.df[column])

	def __str__(self):
		message='Subject ID: '+self.ID+'\n'
		message+='Number of traverses: '+str(len(self.list_dfs))+'\n'
		message+='List_Xs: '+str(self.list_Xs)+'\n'
		message+='List_Ys: '+str(self.list_Ys)+'\n'
		return message

	def read_temp(self, ALL=True, traverse_name=None):
		prefix='energy/temp/'+self.ID+'/'
		list_dfs=[]
		dataFiles = [ f for f in os.listdir(prefix) if f.endswith('.csv') ]
		if ALL==False:
			if traverse_name+'.csv' in dataFiles:
				dataFiles = [traverse_name+'.csv']
		for f in dataFiles:
			df=pd.read_csv(prefix+f)
			TIME=np.array(df['TIME'])
			Weight=np.array(df['Weight'])
			Load=np.array(df['Load'])
			Velocity=np.array(df['Velocity'])
			Slope=np.array(df['Slope'])
			Eta=np.array(df['Eta'])
			Rate=np.array(df['Rate'])
			Fatigue=np.array(df['Fatigue'])

			list_dfs.append(df)
		return list_dfs

	def recombine_features(self,n=1):
		# Returns an array with more features than are the recombination of the old ones.
		X=self.X
		list_Xs=self.list_Xs
		L=len(list_Xs)
		N=L
		if n>self.recombined:
			def rec(j,l):  #Recursive function
				if j==1:
					return l
				lf=[]
				l0=['Weight','Load','Velocity','Slope']
				for e in l:
					index=-1
					for i0 in range(len(l0)):
						if e[0]==l0[i0]:
							index=i0
							for i in range(index+1):
								lf+=rec(j-1,[[l0[i]]+e])
				return lf
			# Recombination of features multiplying linearly.
			# WEIGHT=self('Weight')
			# LOAD=self('Load')
			# VELOCITY=self('Velocity')
			# SLOPE=self('Slope')
			for i in range(self.recombined+1,n+1):
				lf=rec(i,[['Weight'],['Load'],['Velocity'],['Slope']])
				for e in lf:
					# print(e)
					aux=np.ones(len(self[e[0]]))
					new_column_name=str(e[0])
					for j in range(len(e)):
						var=e[j]
						aux=aux*np.array(self[var])
						if j>0:
							new_column_name+='*'+var
					X=np.insert(X, N, aux, axis=1)
					self.df[new_column_name]=aux
					#recombination=('*').join(e)
					list_Xs.append(new_column_name)
					N+=1
			self.recombined=n
			#print(f'Recombined features to n = %d'%self.recombined)
		else: 
			#print(f'Already recombined features to n = %d'%self.recombined)
			pass
		if '1' not in list_Xs:
			X=np.insert(X, 0, 1, axis=1)
			list_Xs=['1']+list_Xs
			self.df['1']=1
		self.X=X
		self.list_Xs=list_Xs			
				

def generate_traverses(num_iters=100, input_model='PL', plot_it=False, ID=False, env=None, save=True):
	#os.mkdir('traverse/test')
	prefix=''
	if not ID: #if there is no subject ID, a new one is created
		ID=str(len(os.listdir('traverse/temp')))  
	ID=str(ID)
	prefix='temp/'+ID+'/'
	if save==True:
		os.mkdir('traverse/'+prefix)
		os.mkdir('energy/'+prefix)

	gravity=9.8
	eta=1.0
	weight=np.random.uniform(50,90)
	if env:  # if there is an environment
		gravity=env.gravity
		eta=env.eta
		if env.weight:
			weight=env.weight
		if env.load:
			load=env.load
	
	for i in range(num_iters):
		load=np.random.uniform(0,30)
		if env:
			if env.load:
				load=env.load

		precision=60   # Time between samples
		total_time=7200 # Total time in Seconds
		number_of_samples=int(total_time/precision)
		velocity=np.random.uniform(0,5)  # Constant speed between 0 and 5 m/s
		xp=[velocity*precision*k for k in range(number_of_samples)] # ASDF
		yp=[500]  	# Starting altitude
		S=np.random.uniform(-20,20)  # Starting Slope, between -20% and 20%
		for k,x in enumerate(xp[:-1]):
			dx=xp[k+1]-x
			yp.append(yp[k]+S*dx/100.0)
			S=S+np.random.uniform(-S/30-2,-S/30+2)
			if abs(S)>20:
				S=20*np.sign(S) # Cap of Slope
		l_points=[xp,yp] 
		if plot_it:
			import matplotlib.pyplot as plt
			plt.plot(xp,yp)
			plt.xlabel('Distance [m]')
			plt.ylabel('Height [m]')
			plt.grid()
			plt.title('Traverse example 6')#+str(i+1))
			plt.xlim(0,xp[-1])
			plt.ylim(min(yp)/1.2,1.2*max(yp))
			plt.show()

		if save==True:
			write_traverse(l_points, filename = prefix+str(i), velocity=velocity, weight=weight, load=load, precision=precision, eta=eta, gravity=gravity)
			add_energy(filename_input = prefix+str(i), filename_output=prefix+str(i), input_model=input_model)	
	return ID
		

#def generate_

if __name__ == '__main__':
	generate_traverses(num_iters=5, plot_it=True, save=False)  # To showcase that the code runs
