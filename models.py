# This script contains different models for metabolic rate
# Metric System Units used: Energies: J, Power: W = J/s, Time: s
# Weights: kg, Speeds: m/s, Slopes: %

import numpy as np 
import scipy
import os
import sys
import pandas as pd
import time

def cal2jul(cal):
	return 4.184*cal

def btu_to_watts(btu):
	return btu*0.29307

# MODELS
# W: body height, L: external load, V: velocity, S: slope %
# Output: Metabolic Rate W = J/s
def GG(W, L, V=0.0, S=0.0, eta=1.0, g=9.8):
	# Givoni-Goldman model, 1971
	return (g/9.8)*eta*(W+L)*(2.3+0.32*max(V*3.6-2.5,0)**1.65+S*(0.2+0.07*max(V*3.6-2.5,0)))*4184/3600


def GG_running(W, L=0.0, V=0.0, S=0.0, eta=1.0, g=9.8):
	mr = GG(W, L, V, S, eta, g)
	return (mr+0.47*(900*4184/3600 - mr))*(1+S/100) 


def PL(W, L=0.0, V=0.0, S=0.0, eta=1.0, g=9.8):
	return 1.5*W+2.0*(W+L)*((L/W)**2)+eta*(W+L)*(1.5*V**2+0.35*V*S)

def PL_santee(W, L=0.0, V=0.0, S=0.0, eta=1.0, g=9.8):
	mr = PL(W, L, V, S, eta, g)    #PANDOLF EQ
	c = eta*( (S*(W+L)*V )/3.5 - (((W+L)*(S+6)**2)/W) +(25-V**2))
	delta=0.2
	if S>=delta:
		return mr
	elif S<-delta:
		return mr-c
	else:
		return (mr*(S+delta)+(mr-c)*(delta-S))/(2*delta)   #smooth


def ACSM(W, L=0.0, V=0.0, S=0.0, eta=1.0, g=9.8):
	return (0.1*V/60+1.8*V*S/60+3.5)*20.9*(1.0/60)

def SANTEE(W, L=0.0, V=0.0, S=0.0, eta=1.0, g=9.8):
	alpha=np.arctan(S/100)  # angle
	W_level = (3.28 * (W+L) + 71.1) * (0.661* V *np.cos(alpha) + 0.115)
	if alpha>0:
		W_slope=3.5*(W+L)*g*V*np.sin(alpha)
	elif alpha<=0:
		W_slope=2.4*(W+L)*g*V*np.sin(alpha)*0.3**(abs(alpha)/7.65)
	return W_level+W_slope

def EE(W, L=0.0, V=0.0, S=0.0, eta=1.0, g=9.8):
	W_level = (g/9.8)*(1.44+1.94*V**0.43+0.24*V**4)
	W_slope = 0.34*V*S*(1-1.05**(1-1.1**(S+32)))
	return (W+L)*(W_level+W_slope)


MODELS={'GG':GG, 'PL':PL, 'ACSM':ACSM, 'SANTEE':SANTEE, 'GG_running':GG_running, 'PL_santee':PL_santee, 'EE':EE}


def metabolic_rate_to_O2(mr):
	O2 = mr*5/0.0143
	return O2


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
	if type(velocity)==float or type(velocity)==int: #IF the velocity is not a Vector, but a single number
		meters_per_unit_of_time = precision*velocity
		x = np.array([j*meters_per_unit_of_time for j in range(int(xp[-1]/meters_per_unit_of_time)+1)])
	else:
		x = np.cumsum(precision*velocity)
	TIME=np.array([j*precision for j in range(len(x))])	
	# x: meters traveled on each timestamp
	y = np.interp(x, xp, yp, left=yp[0], right=yp[-1])
	aux = 100*(y[1:]-y[:-1])/(x[1:]-x[:-1]) # derivative # %percentage
	Slope=np.zeros(len(TIME))
	Slope[0]=aux[0]
	Slope[-1]=aux[-1]
	for i in range(1,len(Slope)-1):
		Slope[i]=0.5*(aux[i-1]+aux[i])

	df = pd.DataFrame({	'TIME': TIME,
						'X':x,
						'Y':y,
						'Weight': weight,
                    	'Load': load,
                    	'Velocity': velocity,
                    	'Slope': Slope,
                    	'Eta': eta,
                    	'Gravity': gravity})
	
	if filename=='':
		filename=str(float(time.time()))

	df.to_csv('traverse/'+ filename + '.csv', index=False, columns=['TIME','X','Y','Weight','Load','Velocity','Slope','Eta','Gravity'])
	
	# handle=open('traverse/'+ filename + '.csv', 'w')
	# handle.write(df.to_csv(index=False,columns=['TIME','Weight','Load','Velocity','Slope','Eta','Gravity']))
	# handle.close()	

	return True


# 	DATAFRAMES:
#	TIME, RATE, Fatigue, Weight, Load, Velocity, Slope, Eta, Gravity


def add_energy(filename_input,filename_output,input_model=0, noise=0.0):
	model=input_model
	if input_model in MODELS:  # if input_model is a string identifier
		model=MODELS[input_model]
	if input_model==0:
		model=PL   # By defect we choose PL
	df=pd.read_csv('traverse/'+filename_input+'.csv')
	TIME=np.array(df['TIME'])
	Weight=np.array(df['Weight'])
	Load=np.array(df['Load'])
	Velocity=np.array(df['Velocity'])
	Slope=np.array(df['Slope'])
	Eta=np.array(df['Eta'])
	Gravity=np.array(df['Gravity'])
	

	Rate=np.array(list(map(model,Weight,Load,Velocity,Slope,Eta,Gravity)))     # model can be anything
	if noise!=0.0:
		Rate *= (np.random.normal(loc=1.0, scale=noise, size=len(Rate)))
	
	Fatigue=np.zeros(len(Rate))
	for i, instant_rate in enumerate(Rate[:-1]):
		Fatigue[i+1]=Fatigue[i]+instant_rate*(TIME[i+1]-TIME[i])
	df['Rate']=Rate
	df['Fatigue']=Fatigue


	df.to_csv('energy/'+ filename_output + '.csv', index=False, columns=['TIME','Weight','Load','Velocity','Slope','Eta','Gravity','Rate','Fatigue'])

