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
    vocablist = []
	for word in dataset:
		if word not in vocablist:
			vocablist.extend(word)
	return vocablist

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
函数说明:朴素贝叶斯分类器训练函数 条件概率
Parameters:
	trainmat - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	traincat - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0vect - 非的条件概率数组
	p1vect - 侮辱类的条件概率数组
	pabusive - 文档属于侮辱类的概率
p1num is:
 [0. 2. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 1. 3. 0. 0. 0. 0. 0. 0. 0. 0. 1.
 1. 0. 2. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0.]
p1denom: 19.0
[0.         0.10526316 0.         0.         0.         0.
 0.         0.05263158 0.05263158 0.05263158 0.05263158 0.05263158
 0.         0.05263158 0.15789474 0.         0.         0.
 0.         0.         0.         0.         0.         0.05263158
 0.05263158 0.         0.10526316 0.05263158 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.05263158 0.05263158 0.         0.         0.05263158
 0.        ] 
 '''
def naivebayes0(trainmat, traincat):
	nummat = len(trainmat)
  	pabusive = sum(traincat)/float(nummat)#文档属于侮辱类的概率
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
'''
根据李航的《统计学习方法》，上面的估计概率的 方法叫做————“极大似然估计”
有两个缺点：
1，有些概率=0，那么，后面的概率相乘，会影响计算结果
2,很多小数相乘，会下溢出，所以加上log
laplace smoothing
reference: https://github.com/TingNie/Machine-learning-in-action/blob/master/bayes/naiveBayes.ipynb
'''
def naivebayes1(trainmat, traincat):
	nummat = len(trainmat)
  	pabusive = sum(traincat)/float(nummat)#文档属于侮辱类的概率
  	numword = len(trainmat[0])
  	p0num = np.ones(numword); p1num = np.ones(numword)
  	p0denom = 2.00 ; p1denom = 2.00
	for i in range(nummat):
		if traincat[i] == 1:#在侮辱类文档中，统计词频，最高的更有可能是侮辱类的词,条件概率
	  		p1num += trainmat[i]
		  	p1denom += sum(trainmat[i])
	  	else:
		  	p0num += trainmat[i]
		  	p0denom += sum(trainmat[i])
	p0vec = np.log(p0num/float(p0denom))
	p1vec = np.log(p1num/float(p1denom))
	return p0vec,p1vec,pabusive				  
"""
Parameters:
	test - 待分类的词条数组
	p0vec - 非侮辱类的条件概率数组
	p1vec -侮辱类的条件概率数组
	pabusive - 文档属于侮辱类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
"""
def classify(test, p0Vec, p1Vec, pabusive):
  	p1 = sum(testVec * p1Vec) + np.log(pabusive)    #因为是log,所以这里是求和以及+号操作
    #p(y=1/w) = p(w/y=1) * p(y=1),注意这里需要乘上，testVec,过滤掉那些为0的特征的概率
    p0 = sum(testVec * p0Vec) + np.log(1.0 - pabusive) #p(y=0) = 1 - p(y=1)
    if p1 > p0: #选取最大的概率的类
        return 1
    else: 
        return 0

if __name__ == '__main__':
    postingList,classVec = loadDataSet()
    vocabularyList = createlist(postingList)
    print ("the vocabulary list is:\n",vocabularyList)
    returnVec = setofwords2vec(vocabularyList,postingList[0])
    print ("post0 vector=\n",returnVec)
    trainVec = []
    for post in postingList:
        trainVec.append(setofwords2vec(vocabularyList,post))
    print("all post vector are:\n",trainVec)
    p0Vect,p1Vect,pab = naivebayes0(trainVec,classVec)
    p0,p1,pa =naivebayes1(trainVec,classVec)
    test0 = ['love', 'my', 'dalmation']
    testVec0 = setofwords2vec(vocabularyList, test0)
    print (test0,'classified as: ',classify(testVec0,p0,p1,pa))
    test1 = ['stupid', 'garbage']
    testVec1 = setofwords2vec(vocabularyList, test1)
    print (test1,'classified as: ',classify(testVec1,p0,p1,pa))
			      
import re			      
'''
函数说明:接收一个大字符串并将其解析为字符串列表,小写单词
'''
def textparse(bigstring):
  	text = re.split(r'\W+',bigstring) ##将特殊符号作为切分标志进行字符串切分，即非字母、非数字https://docs.python.org/3/library/re.html
  	return [words.lower() for words in text if len(words) > 2]   #除了单个字母，例如大写的I，其它单词变成小写
'''
函数说明:测试朴素贝叶斯分类器
过滤垃圾邮件
步骤1:数据准备
将ham和spam里的邮件分别读出来，标记，生成vocablist和classvec,要用到的函数有textparse（bigstring）（切割单词），createlist（dataset）
步骤2:划分数据集
取任意40个邮件作为训练集，10个作为测试集，使用random任意生成
步骤3:生成矩阵
使用函数setofwords2vec(vocablist, inputset)，其中inputset需要遍历每一封邮件
步骤4:计算概率
步骤5:统计错误率
'''

import random
def spamTest():
    wordlist=[];classvec=[]
    for i in range(1,26):
        word = textparse(open('email/spam/%d.txt' % i,errors='ignore').read())
        wordlist.append(word)
        classvec.append(1)
        word = textparse(open('email/ham/%d.txt' % i,errors='ignore').read())
        wordlist.append(word)
        classvec.append(0)
    vocablist = createlist(wordlist)
    trainindex = list(range(50)); testindex=[]
    for i in range(10):
        testi=int(random.uniform(0,len(trainindex)))
        testindex.append(testi)  		
        del(trainindex[testi])
    trainmat=[];traincat=[]
    for itrain in trainindex:
        trainmat.append(setofwords2vec(vocablist, wordlist[itrain]))
        traincat.append(classvec[itrain])
    p0,p1,pa = naivebayes1(trainmat, traincat)
    error = 0
    for itest in testindex:
        testvec = setofwords2vec(vocablist, wordlist[itest])	  	
        if classify(testvec, p0, p1, pa) != classvec[itest]:
            error+=1
            print("classification error",wordlist[itest])
    print ('the error rate is: ',float(error)/len(testindex))
