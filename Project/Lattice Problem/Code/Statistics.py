# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:27:38 2018

@author: Mikkel
"""

import numpy as np
import matplotlib.pyplot as plt
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

col = ['Function','GridSize','Seconds']
data = read_csv("benchmark_results14_52.csv")


data = (pd.DataFrame(data, columns = col))
dataAvg = data.groupby(['Function','GridSize'])['Seconds'].mean().reset_index()
#dataSmall = data[data['Target'] == 'small']
#dataMedium = data[data['Target'] == 'medium']
#dataBig = data[data['Target'] == 'big']
#dataSizeAverages = data.groupby(['Device', 'Target'])['PlacingTime'].mean().reset_index()

print("data averages:")
#print(dataSizeAverages)

print("Plot for average placing times")
sns.barplot(x='GridSize', y='Seconds', data=data)
plt.legend(loc=(1.04,0))
plt.ylabel("Placing time, seconds")
plt.show()

print("Plot for average placing times, by target")
sns.barplot(x='GridSize', y='Seconds', data=data, hue='Function')
plt.legend(loc=(1.04,0))
plt.ylabel("Placing time, seconds")
plt.show()

#dataAvg[dataAvg['Device'] == 'TrickTracer']['PlacingTime']

cpuX = dataAvg[dataAvg['Function'] == 'LoopDistributedCPU']['GridSize']
cpuY = dataAvg[dataAvg['Function'] == 'LoopDistributedCPU']['Seconds']
gpuX = dataAvg[dataAvg['Function'] == 'LoopDistributedGPU']['GridSize']
gpuY = dataAvg[dataAvg['Function'] == 'LoopDistributedGPU']['Seconds']

print("Plot for average placing times, by target")
plt.plot(cpuX, cpuY)
plt.plot(gpuX, gpuY)
plt.legend(loc=(1.04,0))
plt.ylabel("Placing time, seconds")
plt.show()

print("GridSize  CPU   GPU")
for i in range (0, np.size(cpuY)):
    print(str(cpuX.values[i]) + "      " + str(cpuY.values[i]) + "   " + str(gpuY.values[i]))



speedup = []
print("Speedup factor:")
print("GridSize  factor")
for i in range (0, np.size(cpuY)):
    speedup.append(cpuY.values[i]/gpuY.values[i])
    print(str(cpuX.values[i]) + "    " + str(cpuY.values[i]/gpuY.values[i]))

sns.barplot(cpuX, speedup)
plt.legend(loc=(1.04,0))
plt.ylabel("Speedup factor")
plt.show()

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
