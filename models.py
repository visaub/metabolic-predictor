# This script contains different models for metabolic rate
# Metric System Units used: Energies: J, Power: W = J/s, Time: s
# Weights: kg, Speeds: m/s, Slopes: %

import numpy as np 
import scipy
import os
import sys
import pandas as pd
import time

#pd.core.frame.DataFrame


def cal2jul(cal):
	return 4.184*cal

# MODELS
# W: body height, L: external load, V: velocity, S: slope %
# Output: Metabolic Rate W = J/s
def GG(W,L,V=0.0, S=0.0, eta=1.0, g=9.8):
	return eta*(W+L)*(2.3+0.32*max(V*3.6-2.5,0)**1.65+S*(0.2+0.07*max(V*3.6-2.5,0)))*4184/3600

def PL(W,L=0.0, V=0.0, S=0.0, eta=1.0, g=9.8):
	return 1.5*W+2.0*(W+L)*((L/W)**2)+eta*(W+L)*(1.5*V**2+0.35*V*S)

def ACSM(W,L,V=0.0,S=0.0, eta=1.0, g=9.8):
	return (0.1*V/60+1.8*V*S/60+3.5)*20.9*(1.0/60)

def SANTEE(W,L,V=0.0,S=0.0, eta=1.0, g=9.8):
	alpha=np.arctan(S/100)
	W_level=(3.28*(W+L)+71.1)*(0.661*V*np.cos(alpha)+0.115)
	if alpha>0:
		W_slope=3.5*(W+L)*g*V*np.sin(alpha)
	elif alpha<=0:
		W_slope=2.4*(W+L)*g*V*np.sin(alpha)*0.3**(abs(alpha)/7.65)
	return W_level+W_slope

MODELS={'GG':GG, 'PL':PL, 'ACSM':ACSM, 'SANTEE':SANTEE}


def speed_astronaut(S=0.0):
	# alpha=np.arctan(S/100)
	if S<-20: return 0.05
	elif S<-10: return 0.095*S+1.95
	elif S<0: return 0.06*S+1.6
	elif S<6: return -0.2*S+1.6
	elif S<15: return -0.039*S+0.634
	elif S>=15: return 0.05

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	S=np.linspace(-20,20,1000)
	MR=np.zeros(1000)
	for i,s in enumerate(S):
		alpha=np.arctan(s/100)
		MR[i]=SANTEE(120,0,speed_astronaut(s),g=1.6)/(speed_astronaut(s)*np.cos(alpha))
	plt.plot(S,MR)
	plt.grid()
	plt.show()




def write_traverse(l_points, filename='', velocity=1.25, weight=80, load=0, precision=60, eta=1.0, gravity=9.8):
	"""write_traverse(l_points, filename='', velocity=1.25, weight=80, load=0, precision=60, eta=1.0)"""
	# 1.25 m/s == 4.5 km/h. velocity could be a vector.  precision: 60 seconds of unit of time
	xp = np.array(l_points[0])
	yp = np.array(l_points[1])
	if type(velocity)==type(1.0) or type(velocity)==type(1): #Velocity either a float or integer
		meters_per_unit_of_time = precision*velocity
		x = np.array([j*meters_per_unit_of_time for j in range(int(xp[-1]/meters_per_unit_of_time)+1)])
	else:
		x = np.cumsum(precision*velocity)
	Time=np.array([j*precision for j in range(len(x))])	
	# x: meters traveled on each timestamp
	y = np.interp(x, xp, yp, left=yp[0], right=yp[-1])
	aux = 100*(y[1:]-y[:-1])/(x[1:]-x[:-1]) # derivative # %percentage
	Slope=np.zeros(len(Time))
	Slope[0]=aux[0]
	Slope[-1]=aux[-1]
	for i in range(1,len(Slope)-1):
		Slope[i]=0.5*(aux[i-1]+aux[i])

	df = pd.DataFrame({	'TIME': Time,
						'X':x,
						'Y':y,
						'Weight': weight,
                    	'Load': load,
                    	'Velocity': velocity,
                    	'Slope': Slope,
                    	'Eta': eta,
                    	'Gravity': gravity})
	
	if filename=='':
		filename=str(int(time.time()))

	df.to_csv('traverse/'+ filename + '.csv', index=False, columns=['TIME','X','Y','Weight','Load','Velocity','Slope','Eta','Gravity'])
	
	# handle=open('traverse/'+ filename + '.csv', 'w')
	# handle.write(df.to_csv(index=False,columns=['TIME','Weight','Load','Velocity','Slope','Eta','Gravity']))
	# handle.close()	

	return True


# 	DATAFRAMES:
#	TIME, RATE, Fatigue, Weight, Load, Velocity, Slope, Eta, Gravity


def add_energy(filename_input,filename_output,input_model=0):
	model=input_model
	if input_model in MODELS:  # if input_model is a string identifier
		model=MODELS[input_model]
	if input_model==0:
		model=PL
	df=pd.read_csv('traverse/'+filename_input+'.csv')
	TIME=np.array(df['TIME'])
	Weight=np.array(df['Weight'])
	Load=np.array(df['Load'])
	Velocity=np.array(df['Velocity'])
	Slope=np.array(df['Slope'])
	Eta=np.array(df['Eta'])
	Gravity=np.array(df['Gravity'])
	
	Rate=np.array(list(map(model,Weight,Load,Velocity,Slope,Eta,Gravity)))
	
	Fatigue=np.zeros(len(Rate))
	for i, instant_rate in enumerate(Rate[:-1]):
		Fatigue[i+1]=Fatigue[i]+instant_rate*(TIME[i+1]-TIME[i])
	df['Rate']=Rate
	df['Fatigue']=Fatigue


	df.to_csv('energy/'+ filename_output + '.csv', index=False, columns=['TIME','Weight','Load','Velocity','Slope','Eta','Gravity','Rate','Fatigue'])

	# handle=open('energy/'+ filename_output + '.csv', 'w')
	# handle.write(df.to_csv(index=False))
	# handle.close()

""" Writing other functions
def loadAllData():
    d = {}
    dataFiles = [ x for x in os.listdir('data') if x.endswith('.csv') ]
    for f in dataFiles:
        print ('reading file: %s' % f)
        symbol = f.split('.')[0]
        df = pd.read_csv( 'data/' + f )
        df['Date'] = [ datetime.strptime( x, '%m/%d/%y' ) for x in df['Date'] ]
        df.index = df['Date']
        print (df.head())
        d[ symbol ] = df
        print ('read %s' % f)
    return d
 """