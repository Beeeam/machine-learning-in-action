import numpy as np
import operator
import collections

#创建数据集
def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    #作为基准，在之后我们需要计算输入的数据与group之间的距离
    label = ['A','A','B','B']
    #分类标签
    return group,label

#分类器
#Parameters:
	#inX - 需要分类的数据
	#dataSet - 分类基准，即上面的group
	#labels - 分类标签
	#k - kNN算法参数,选择距离最小的k个点
  #reference：GitHub（https://github.com/Jack-Cherish/Machine-Learning/blob/master/kNN/1.%E7%AE%80%E5%8D%95k-NN/kNN_test01.py）

def classify0(inX, dataSet, labels, k):
    #the code from book
    #dataSetsize = dataSet.shape[0]   # 告诉我们dataset里有多少个元素
    #diffMat = tile(inX, (dataSetsize,1))-dataSet   #tile函数将inX复制相应的次数
    #sqDiffMat = diffMat**2
    #sqDistances = sqDiffMat.sum(axis=1)# axis=1则输出的结果是一个数组，分别是每一项的值，如果default输出一个int，是所有值的求和
    #distances = sqDistances**0.5
    #实际上不需要将inX复制成相同数量的array，直接相减即可，得到的值就是by coordinate计算的
    dist = np.sum((inX-dataSet)**2,axis=1)**0.5 #输出的值是[1,2,3,4]分别对应着输入值与相应的基准的距离
    sortedDistIndicies = dist.argsort() #按照dist这个list中的值从小到大排列，分别给出相应的顺序，例如[2,4,1,3]则结果是[2,0,3,1]
    classcount = {}
    for i in range(k): #dataset共有4个数则k一定要小于4，否则无法判断
        votelabel = labels[sortedDistIndicies(i)] #会得到一个list，['B','A','B']
        classcount[votelabel] = classcount.get(votelabel,0)+1 #会得到一个dictionary{'B':2 ;'A':1 },.get('B',0)意味着取‘B’的值，如果值不存在则输出0
    #sortedclasscount = sorted(classcount.items(),key = operater.itemgetter(1), reverse = True) #将dictionary的item取出来,以dict的形式储存，py3中没有iteritems这个attribute，按照dictionary的内容进行排序
    #return sortedclasscount[0][0]    
    #我们可以利用collections里的counter函数进行统计 
    label = collections.Counter(votelabel).most_common(1)[0][0] #按照votelabel的多少进行计数，并按照从大到小排列，格式为dict，most_common(1)只输出一个
	  return label 

if __name__ == '__main__':
	#创建数据集
	group, labels = createDataSet()
	#需要分类的数据
	test = [0,0]
	#kNN分类
	test_class = classify0(test, group, labels, 3)
	#打印分类结果
	print(test_class)
