import numpy as np
def loaddata(filename):
	datamat=[];labelmat=[]
	fr = open(filename)
	m = len(fr)
	for line in fr:
		linearr = line.stripe().split('\t')
		datamat.append(float(linearr[:m-1]))
		labelmat.append(float(linearr[-1]))
	return datamat,labelmat
		
