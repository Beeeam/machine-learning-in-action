import numpy as np
'''
postinglist里有stupid这样的词，则classvec是1
'''
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
''' 
将dataset中的词汇提出做成wordlist，不重复
'''
def createlist(dataset):
    wordlist = []
	for word in dataset:
		if word not in wordlist:
			wordlist.extend(word)
	return wordlist

'''
根据inputset将vocablist向量化，在inputset中的记为1，否则记为0
'''
def setofwords2vec(vocablist, inputset):
	returnvec = [0]*len(vocablist)
	for word in inputset:
		if word in vocablist:
			returnvec[vocablist.index(word)] = 1
		else:
			print(("the word: %s is not in my Vocabulary!" % word)
	return returnvec
'''
函数说明:朴素贝叶斯分类器训练函数
Parameters:
	trainmat - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	traincat - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0vect - 非的条件概率数组
	p1vect - 侮辱类的条件概率数组
	pabusive - 文档属于侮辱类的概率
'''
def naivebayes0(trainmat, traincat):
	nummat = len(trainmat)
  	pabusive = sum(traincat)/float(numofmat)#文档属于侮辱类的概率
  	numword = len(trainmat[0])
  	p0num = np.zeros(numword); p1num = np.zeros(numword)
  	p0denom = 0.00 ; p1denom = 0.00
	for i in range(nummat):
		if traincat[i] == 1:#在侮辱类文档中，统计词频，最高的更有可能是侮辱类的词
	  		p1num += trainmat[i]
		  	p1denom += sum(trainmat[i])
	  	else:
		  	p0num += trainmat[i]
		  	p0denom += sum(trainmat[i])
	p0vec = p0num/float(p0denom)
	p1vec = p1num/float(p1denom)
	return p0vec,p1vec,pabusive
				  
				  
				  
				  
