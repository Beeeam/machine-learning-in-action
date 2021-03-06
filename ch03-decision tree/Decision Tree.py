import math
import operator
import pickle
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
		
'''
每个节点是一个特征，一个特征有多个值，如果标签更多地集中在一个值，这个标签的熵越小，infogain越大，则说明这个特征能更好地反应这个标签
需要遍历每一个特征，并对特征中所有的值求熵
寻找最好的特征
'''
def bestfeat(dataset):
	featnum = len(dataset[0])-1
	baseent = entropy(dataset)
	bestinfogain = 0
	bestfeature = 0
	for featvec in dataset:
		for i in range(featnum):
			#统计有多少个值,set函数是把一个list/dict/tuple中重复元素去掉
			featlist = [example[i] for example in dataset]
			uniqval = set(featlist)
			newent = 0
			for value in uniqval:
				subdata = datasplit(dataset, i, value)
				prob = len(subdata)/float(len(dataset))
				newent += prob*entropy(subdata)
			infogain = baseent - newent
			if (infogain > bestinfogain):
				bestinfogain = infogain
				bestfeature = i
		return beatfeature
'''
统计结果标签中最多的 'yes' 'no'哪个多就分类哪个,返回的是类别
'''
def classcount(classlist):
	count = {}
	for i in classlist:
		if i not in count.keys():
			count[i] = 0
		count[i] += 1
	sortedcount = sort(count.items(),key=operator.itemgetter(1), reverse = True)
	return sortedcount[0][0]
'''
创建决策树，得到决策树{'house holding': {0: {'employed': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
'''
def createtree(dataset, feats):
    classlist =[example[-1] for example in dataset]
    if len(set(classlist)) == 1:#当所有的标签‘yes’ or ‘no’都相同时，返回这个标签
        return classlist[0]
    if len(dataset[0]) == 1:#当被删的只剩一个的时候（所有特征都被用完了），返回最多的标签
        return classcount(classlist)
    best = bestfeat(dataset)
    featname = feats[best]
    tree = {featname:{}}
    del(feats[best])
    val = [value[best] for value in dataset]
    uniqval = set(val)
    for i in uniqval:
        subset = datasplit(dataset, best, i)
        subfeats = feats[:]#保留原有的featurename 不被修改
        tree[featname][i] = createtree(subset, subfeats)
    return tree	
'''
testvec需要分类的features inputtree 决策树 labels 特征标签
从bestfeatures的值开始判断 如果不是最佳的继续分类
'''
def classify(inputtree, labels):
    testvec = []
    for i in range(len(labels)-1):
        vec = input('%s:' %labels[i])
        testvec.extend(vec)
    def clas(inputtree, labels, testvec):
        firststr = next(iter(inputtree.keys()))#树中第一个feature，即best feature
        seconddict = inputtree[firststr]#第二个tree{0: {'employed': {0: 'no', 1: 'yes'}}, 1: 'yes'}
        labelindex = labels.index(firststr)#feature在labels的index
        key = int(testvec[labelindex])#相应feature在testvec中的值作为键
        dictval = seconddict[key]#取相应键的值
        if isinstance(dictval, dict):
            result = clas(dictval, labels, testvec)
        else:
            result = dictval
		return result
    return clas(inputtree, labels, testvec)			
"""
函数说明:存储决策树
Parameters:
	inputTree - 已经生成的决策树
	filename - 决策树的存储文件名
""" 
def storeTree(inputTree, filename):
	with open(filename, 'wb') as fw:
		pickle.dump(inputTree, fw)

"""
函数说明:读取决策树
Parameters:
	filename - 决策树的存储文件名
Returns:
	pickle.load(fr) - 决策树字典
""" 
def grabTree(filename):
	fr = open(filename, 'rb')
	return pickle.load(fr)        

if __name__ == '__main__':
    dataset,labels = createDataSet()
    tree = createtree(dataset, labels)
    dataset,labels = createDataSet()
    test=[1,0,1]
    print(dataset)
    print(labels)
    print(classify(tree, labels))
