import matplotlib.pyplot as plt
import numpy as np
import operator
import collections

#分类器函数
def classify0(inX, dataset, labels, k):
    dist = np.sum((inX-dataset)**2,axis=1)**0.5
    votelabel = []
    for i in dist.argsort().[:k]:
        votelabel.append(labels[i])
    label = collections.Counter(votelabel).most_common(1)[0][0]
    return label

#读取文件 返回dataset以及labels largedoses=3 smalldoses=2 didntlike=1
def file2matrix(filename):
	dataset = []
	labels = []
	fr = open(filename)
	for line in fr.readlines():
		line = line.strip().split(‘\t')#将每行的内容切分成一个数组[40920, 8.326976, 0.953952, ''largeDoses'']
		dataset.append(line[:3]) #知道数据只有4列因此可以直接用3 否则应该有 linenumber=len(line)
		if line[-1] == 'largeDoses':
			labels.append(3)
		elif line[-1] == 'smallDoses':
			labels.append(2)
		else:
			labels.append(1)
	return np.array(dataset,dtype=np.float64), np.array(labels)# this is very important to have dtype=np.float64 otherwise we can not get the right figur

#visualization the data 以matrix中两列数据为x，y轴，喜欢程度用颜色区分
#reference:https://github.com/Jack-Cherish/Machine-Learning/blob/master/kNN/2.%E6%B5%B7%E4%BC%A6%E7%BA%A6%E4%BC%9A/kNN_test02.py
def plot(dataset, labels):
	#将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
	#当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
	fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))
	#画出散点图,以dataset矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据		
	LabelsColors = []	
	for i in labels:
		if i == 1:
			LabelsColors.append('black')
		if i == 2:
			LabelsColors.append('orange')
		if i == 3:
			LabelsColors.append('red')
	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][0].scatter(x=dataset[:,0], y=dataset[:,1], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs[0][0].set_title('pliot distance and game time')
	axs[0][0].set_xlabel('pliot distance')
	axs[0][0].set_ylabel('game time') 
	#设置标题,x轴label,y轴label
	axs[0][1].set_title('pliot distance and ice cream consumption')
	axs[0][1].set_xlabel('pliot distance')
	axs[0][1].set_ylabel('ice cream consumption') 
	#画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰淇淋)数据画散点数据,散点大小为15,透明度为0.5
	axs[1][0].scatter(x=dataset[:,1], y=dataset[:,2], color=LabelsColors,s=15, alpha=.5, label=Labelslabels)
	#设置标题,x轴label,y轴label
	axs[1][0].set_title('game time and ice cream consumption')
	axs[1][0].set_xlabel('game time')
	axs[1][0].set_ylabel('ice cream consumption') 				
	#设置图例
	didntLike = plt.Line2D([], [], color='black', marker='.',
				markersize=6, label='didntLike')
	smallDoses = plt.Line2D([], [], color='orange', marker='.',
				markersize=6, label='smallDoses')
	largeDoses = plt.Line2D([], [], color='red', marker='.',
				markersize=6, label='largeDoses')
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
	plt.show()	
								  
#normalization 返回归一化后的矩阵，极差，最小值
def normaldata(dataset):
	min_data = dataset.min(0)	
	max_data = dataset.max(0)
	ranges = max_data - min_data
	normeddata = (dataset - min_data)/ranges
	return normeddata, ranges, min_data

#分类器测试函数 取数据集中10%作为测试集 验证正确率
def datingclasstest():
	ratio = 0.1 #取前10%作为验证集inX
	datingdata, datinglabels = file2matrix('datingTestSet.txt')
	normed, ranges, min_data = normaldata(datingdata)
	m = normed.shape[0] 
	num = int(m * ratio)
	error = 0
	for i in range(num) :
		result = classify0(normed[i,:], normed[num:m, :],datinglabels[num:m], 4)#m-num项是训练集，即dataset以及labels
		print('the result is: %d, the answer is: %d' % (result, datinglabels[i]))
		if result != datinglabels[i]:
			error += 1
	print('the total error rate is: %f' %(error/float(num)))
                                  
#给一个人分类
def classifyperson():
	resultlist = ['do not', 'a little bit','very']
	pliotdist = float(input('frequent flier miles earned per year:'))
	gametime = float(input('percent of time spent playing video game:'))
	ice = float(input('liters of ice cream consumed per year:'))
	testdata = np.array([pliotdist, gametime, ice])
	datingdata, datinglabels = file2matrix('datingTestSet.txt')
	normeddata, ranges, min_data = normaldata(datingdata)
	normedtest = (testdata-min_data)/ranges #书上没有normalized这个数据，但这个方法有问题如果testdata小于mindata会出现负值
	reslut = classify0(normedtest, normeddata, datinglabels, 4)
	print('you will probably %s like this person.' %(resultlist[reslut-1]))
								  
if __name__ == '__main__':
	datingclasstest()
	classifyperson()
