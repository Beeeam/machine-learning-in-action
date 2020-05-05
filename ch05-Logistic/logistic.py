import numpy as np
import matplotlib.pyplot as plt

'''
sigmoid函数
'''
def sigmoid(inX):
	return 1/(np.exp(-inX)+1)

'''
加载数据，输出数据矩阵以及标签
'''
def loaddata(filename):
	datamat=[];labelmat=[]
	fr = open(filename)
	for line in fr.readlines():
		linearr = line.strip().split('\t')
		datamat.append([1.0,float(linearr[0]),float(linearr[1])])
		labelmat.append(float(linearr[-1]))
	return datamat,labelmat

'''
梯度上升法，迭代次数=500,步长=0.0001
'''
def gradascent(data,label):
    datamat =np.mat(data)
    labelmat = np.mat(label).transpose()#transpose()转置    
	m,n = datamat.shape
    weight = np.ones((n,1))#n行1列的向量
    iters = 500
    learn_rate = 0.001
    for i in range(iters):
        out = sigmoid(datamat*weight)#m行1列的向量
        error = labelmat - out
        weight = weight+learn_rate * datamat.transpose() * error
    return weight.getA()
'''
绘图
'''
datamat,labelmat = loaddata('testSet.txt')
print(datamat[0])
weight=gradascent(datamat,labelmat)
x,y = np.array(datamat),np.array(labelmat)
label0 = np.where(y.ravel()==0)
plt.scatter(x[label0,1],x[label0,2],marker='x',c='r',label=0)
label1 = np.where(y.ravel()==1)
plt.scatter(x[label1,1],x[label1,2],marker='o',c='b',label=1)
xx = np.arange(-4.0,3.0,0.1)
yy = (-weight[0] - weight[1]*xx) / weight[2]
plt.plot(xx,yy,"y_")
plt.legend(loc='upper left')
plt.show()
'''
随机梯度上升法，随机选择样本进行优化，步长随着迭代变短,iters=200
'''
import random
def sga(data,label):
    datamat = np.mat(data)
    labelmat = np.mat(label).transpose()#1行m列
    m,n = datamat.shape
    weight = np.ones((n,1))
    iters = 200
    for i in range(iters):
        dataindex = list(range(m))
        for j in range(m):
            learn_rate = 4/(i+j+1)+0.01
            randinx = int(random.uniform(0,len(dataindex)))
            out = sigmoid(datamat[randinx]*weight)
            error = labelmat[randinx] - out
            weight = weight + learn_rate  *  datamat[randinx].transpose() *error
            del(dataindex[randinx])                                
    return weight.getA()
