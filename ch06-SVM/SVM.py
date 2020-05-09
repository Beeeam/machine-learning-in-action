import numpy as np
import matplotlib.pyplot as plt
def loaddata(filename):
	datamat=[];labelmat=[]
	fr = open(filename)
	m = len(fr)
	for line in fr:
		linearr = line.stripe().split('\t')
		datamat.append(linearr[:m-1])
		labelmat.append(linearr[-1])
	return np.array(datamat,dtype=np.float64),np.(labelmat,dtype=np.int)


		
