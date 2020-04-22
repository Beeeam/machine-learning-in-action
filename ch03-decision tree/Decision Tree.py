import math

#计算数据集的entropy，分别统计数据集中所有的类别的概率，代入信息熵公式
import math
def entropy(dataset):
    num = len(dataset)
    labelscount = {}
    #统计每行中的feature，每行的最后一列，输出是字典{feature1: 30, feature2: 20}
    for featvec in dataset: 
        currentlabel = featvec[-1]
        if currentlabel not in labelscount.keys():
            labelscount[currentlabel] = 0
        else:
            labelscount[currentlabel] += 1
    entropy = 0
    #计算信息熵，遍历每个feature
    for key in labelscount:
        prob = float(labelscount[key])/num
        entropy -= prob * math.log(prob,2)
    return entropy
'''
	生成数据集 interested in this data 
	reference：https://github.com/Jack-Cherish/Machine-Learning/blob/master/Decision%20Tree/Decision%20Tree.py
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    featureNames = ['no surfacing','flippers'] 
    return dataSet, featureNames

'''
	划分数据集，将数据集中特征隔离出来，为了能在下一步，计算各个特征对信息熵的影响
	dataset 数据集
    i 需要划分的特征
	value 特征的值
	意思是去掉值为value的第i个特征
'''
def datasplit(dataset, i, value):
	newdata=[]
	for featvec in dataset:
		if featvec[i] == value:
			newfeatvec = featvec[:i]
			newfeatvec.extend(featvec[i+1:])
			newdata.append(newfeatvec)
	return newdata
		
	
	
	
	
	
