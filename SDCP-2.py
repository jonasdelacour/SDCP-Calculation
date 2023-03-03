import numpy as np
import scipy.io
import csv
import io
import os
import matplotlib.pyplot as plt

# progress bar
from tqdm.notebook import tqdm as log_progress
from time import sleep
import timeit

def calcSDCP(D, V, Vtot, alpha, numMets, fracMets, SF2, Nfx, minDetectable, fracClon):
    beta = (-np.log(SF2) - 2*alpha)/4
    L = int(round(minDetectable * numMets * fracClon))
    
    sumL = 1 - fracMets # N=0
    
    # Loop over sum from N=1 to N=L
    for N in log_progress(np.arange(1,L+1), miniters=10000000):
        temp = ( fracMets/(N*np.log(L)) * np.power( sum( np.power((1 - np.exp(-alpha*D - beta*np.power(D,2)/Nfx)), N/numMets) * V/Vtot ), numMets) ) 
            
    return sumL


# get directory with differential DVH files:
path = "K:\Group Korreman\Laura K\Rotterdam\iCycle plans\DiffDVH_PTV2_only"
files = os.listdir(path)


# Loop over patients
for file in files:
    if ".csv" in file:
        with open(os.path.join(path, file)) as f:
            reader = csv.reader(f)
            
            DVHs = []
            
            for row in reader:
                vals = []
                for valStr in row:
                    if len(valStr) > 0:
                        vals.append(float(valStr))
                    
                DVHs.append(vals)
                
            D = np.array(DVHs[0])
            V = np.array(DVHs[1])
            
            #prob = calcSDCP(D, V, sum(V), alpha, numMets, fracMets, SF2, Nfx, minDetectable, fracClon)