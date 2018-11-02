# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:27:38 2018

@author: Mikkel
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure
import seaborn as sns
import csv
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import multipletests

#subjects = 4

def read_csv(fname):
   prefix = ""
   with open(prefix+fname, 'rt') as f:
       reader = csv.reader(f, delimiter = ";")
       data = []
       for row in reader:
               row2 = []
               for a in row:
                   if(a.isdigit()):
                       row2.append(float(a))
                   else:
                       row2.append(a)
               data.append(row2)
       return (data)

col = ['FirstKernel','SetIsActive','SecondKernel','SequentialReduce','ComputeBirthEvents','Bursting/NonBurstingEvents','NewInfectionsKernel','PhageDecay','DiffusionAndApplyMovement','SwapAndZeroArrays','UpdateOccupancy','NutrientDiffusion','SwapZero']
colShort = ['FK','SIA','SK','SR','CBE','B/NBE','NIK','PD','DAM','SZA','UO','ND','SZ2']
data = read_csv("kernel_timings_10_19.txt")


data = (pd.DataFrame(data, columns = col))
dataAvg = data.groupby()['MicroSeconds'].mean().reset_index()
#dataAvg = data.mean()
dataDevs = data.std()

dataDevs2 = dataDevs / np.sqrt(2500)
#dataSmall = data[data['Target'] == 'small']
#dataMedium = data[data['Target'] == 'medium']
#dataBig = data[data['Target'] == 'big']
#dataSizeAverages = data.groupby(['Device', 'Target'])['PlacingTime'].mean().reset_index()

print("data averages:")
#print(dataSizeAverages)

#print("Plot for average placing times")
#sns.barplot(x='GridSize', y='Seconds', data=data)
#plt.legend(loc=(1.04,0))
#plt.ylabel("Placing time, seconds")
#plt.show()
#FirstKernel = data['FirstKernel']
#
print("Plot for average placing times, by target")
sns.barplot(colShort, data[col].mean(), data = data)
plt.legend(loc=(1.04,0))
plt.ylabel("Runtime, Microseconds")
plt.show()

#dataAvg[dataAvg['Device'] == 'TrickTracer']['PlacingTime']
#
#cpuX = dataAvg[dataAvg['Function'] == 'LoopDistributedCPU']['GridSize']
#cpuY = dataAvg[dataAvg['Function'] == 'LoopDistributedCPU']['Seconds']
#gpuX = dataAvg[dataAvg['Function'] == 'LoopDistributedGPU']['GridSize']
#gpuY = dataAvg[dataAvg['Function'] == 'LoopDistributedGPU']['Seconds']
#
#print("Plot for average placing times, by target")
#plt.plot(cpuX, cpuY)
#plt.plot(gpuX, gpuY)
#plt.legend(loc=(1.04,0))
#plt.ylabel("Placing time, seconds")
#plt.show()
#
#
#print("Plot for average placing times, by target")
#sns.barplot(x='Target', y='PlacingTime', data=data, hue='Device')
#plt.legend(loc=(1.04,0))
#plt.ylabel("Placing time, seconds")
#plt.show()
#
#
#print("Plot for small target")
#sns.barplot(x='Device', y='PlacingTime', data=dataSmall)
#plt.legend(loc=(1.04,0))
#plt.ylabel("Placing time, seconds")
#plt.show()
#
#print("Plot for medium target")
#sns.barplot(x='Device', y='PlacingTime', data=dataMedium)
#plt.legend(loc=(1.04,0))
#plt.ylabel("Placing time, seconds")
#plt.show()
#
#print("Plot for big target")
#sns.barplot(x='Device', y='PlacingTime', data=dataBig)
#plt.legend(loc=(1.04,0))
#plt.ylabel("Placing time, seconds")
#plt.show()
#
#hololens = dataAvg[dataAvg['Device'] == 'TrickTracer']['PlacingTime']
#projector = dataAvg[dataAvg['Device'] == 'Projector']['PlacingTime']
#hololensSmall = dataSmall[dataSmall['Device'] == 'TrickTracer']['PlacingTime']
#projectorSmall = dataSmall[dataSmall['Device'] == 'Projector']['PlacingTime']
#hololensMedium = dataMedium[dataMedium['Device'] == 'TrickTracer']['PlacingTime']
#projectorMedium = dataMedium[dataMedium['Device'] == 'Projector']['PlacingTime']
#hololensBig = dataBig[dataBig['Device'] == 'TrickTracer']['PlacingTime']
#projectorBig = dataBig[dataBig['Device'] == 'Projector']['PlacingTime']
#
##
#ttestDevice = stats.ttest_rel(hololens, projector)
#ttestSmall = stats.ttest_rel(hololensSmall, projectorSmall)
#ttestMedium = stats.ttest_rel(hololensMedium, projectorMedium)
#ttestBig = stats.ttest_rel(hololensBig, projectorBig)
##    
#print("T-test for device")  
#print(ttestDevice)
##    
#print("T-test for small")  
#print(ttestSmall)
#
#print("T-test for medium")  
#print(ttestMedium)
#
#print("T-test for big")  
#print(ttestBig)
