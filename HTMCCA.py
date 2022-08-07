if(1==1):#库
    #!usr/bin/env python
    #-*- coding: utf-8 -*-
    from skimage.transform import rotate
    from skimage.feature import local_binary_pattern
    from skimage import data, io
    from skimage.color import label2rgb
    import skimage
    from skimage import feature as ft
    import matplotlib.pyplot as plt
    import scipy.io as sio 
    import scipy.io as sio  
    from numpy import *  
    from PIL import Image#图片处理库
    import scipy.io as sio 
    import os
    import sys
    from numpy import * 
    import copy#深拷贝
    import time
    from sklearn import svm
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import time
    import types
    import sys
    import re
    from numpy import *
    import matplotlib.pyplot as plt
    import matplotlib
    import datetime
    import copy#深拷贝
    import matplotlib.pyplot as plt # plt ??Д???
    import matplotlib.image as mpimg # mpimg ??????
    import numpy as np,pandas as pd 
    from sklearn import neighbors
    from sklearn.metrics import confusion_matrix 
    from sklearn.model_selection import GridSearchCV
    #import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.ensemble import RandomForestClassifier
    from numpy import *
    import matplotlib.pyplot as plt
    import matplotlib
    import datetime
    import copy#深拷贝
    import matplotlib.pyplot as plt # plt ??Д???
    import matplotlib.image as mpimg # mpimg ??????
    import numpy as np,pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d import Axes3D 
    import scipy.io as sio  
    import matplotlib.pyplot as plt  
    import numpy as np  
    import copy#深拷贝
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    #import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.cross_decomposition import CCA
    
    from sklearn.decomposition import KernelPCA
    from sklearn import metrics
    from sklearn.manifold import TSNE
    
if(1==1):#全局量
    if(1==1):#全局常量
        ##########全局常量#####################
        # PATH_R2='G:\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\05-01-01_MDTM(ROI)_HOG(10c10&2c2)\\'
        # PATH_R3='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\新建文件夹\\'
        # PATH_R1='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\03-01-01_DMM(ROI)_HOG(10c10&2c2)\\'
        # PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\05-01-01_DMI(ROI)_HOG(10c10&2c2)(3视角)(未去噪)\\'
        
        # PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\99_备份\\20200513\\04-01-01_DTM(双向异性+ROI)_HOG(10c10&2c2)\\'
        # PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\99_备份\\20200513\\04-01-01_DTM(双向异性+ROI)_HOG(10c10&2c2)\\'
        # PATH_R1='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\99_备份\\20200513\\03-01-01_DMM(双向异性+ROI)_HOG(10c10&2c2)\\'
        
        # PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\99_备份\\20200513\\04-01-01_DTM_HOG(ROI)(10c10&2c2)\\'
        # PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\02_UTD\\01_深度数据\\99_备份\\20200513\\03-01-01_DMM_HOG(ROI)(10c10&2c2)\\'
        
        # PATH_W1='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\03_算法开发与改进\\05_matlab代码\\00_temp\\'
        #MSR
        # PATH_R1='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\04-01-01_DTM_HOG(6轴)(ROI)(10c10&2c2)\\'
        # PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\05-01-01_GP3_DMI(ROI)_HOG(10c10&2c2)\\'
        # PATH_R0='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\01_MSR\\01_深度数据\\'
        # PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\03_DHA\\01_深度数据\\04-01-01_DTM_HOG(6轴)(ROI)(10c10&2c2)(未去噪)\\'
        # PATH_R1='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\03_DHA\\01_深度数据\\05-01-01_GP3_DMI(ROI)_HOG(10c10&2c2)(未去噪)(3层)\\'
        # PATH_R0='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\03_DHA\\01_深度数据\\05-01-01_GP3_DMI(ROI)_HOG(10c10&2c2)(未去噪)(3层)\\'
        #UTD
        PATH_R3='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\05_UTDV2\\02_骨骼数据\\'
        PATH_R1='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\05_UTDV2\\01_深度数据\\04-01-01_DTM_HOG(6轴)(ROI)(10c10&2c2)(未去噪)\\'
        PATH_R2='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\05_UTDV2\\01_深度数据\\05-01-01_GP3_DMI(ROI)_HOG(10c10&2c2)(未去噪)\\'
        PATH_R0='G:\\李兴学术1盘\\01_博士生学术区\\01_人体行为识别\\01_人体行为识别数据库与特征\\02_特征\\05_UTDV2\\01_深度数据\\04-01-01_DTM_HOG(6轴)(ROI)(10c10&2c2)(未去噪)\\'
               
    if(1==0):#MSR   
        if(1==1):#全局变量
            MSR_AS=0#0时表示不分库，1分库
            TRAIN_RATIO=2/3#MSR_AS=1时有用
            
            CROSS_SUBJECT=1#1表示主题交叉验证
        
            PCA_NUM1=550
            PCA_NUM2=550
            LOOP_NUM=10#子空间投影，投影矩阵迭代次数
            
            
        if(1==1):#全局设置
            if(MSR_AS==0):
                CLASS_NUM=20
                ALL_SAMPLE_NUM=557
                TRAIN_NUM=int(ALL_SAMPLE_NUM*TRAIN_RATIO)#训练样本数
                print('###########MSR全部样本############')
            if(MSR_AS==1):
                CLASS_NUM=8
                ALL_SAMPLE_NUM=219
                TRAIN_NUM=int(ALL_SAMPLE_NUM*TRAIN_RATIO)#训练样本数
                print('###########MSR-AS1样本############')
            if(MSR_AS==2):
                CLASS_NUM=8
                ALL_SAMPLE_NUM=228
                TRAIN_NUM=int(ALL_SAMPLE_NUM*TRAIN_RATIO)#训练样本数
                print('###########MSR-AS2样本############')
            if(MSR_AS==3):
                CLASS_NUM=8
                ALL_SAMPLE_NUM=222
                TRAIN_NUM=int(ALL_SAMPLE_NUM*TRAIN_RATIO)#训练样本数
                TEST_NUM=ALL_SAMPLE_NUM-TRAIN_NUM#测试样本数
                print('###########MSR-AS3样本############')
            TEST_NUM=ALL_SAMPLE_NUM-TRAIN_NUM#测试样本数
    if(1==0):#UTD
        if(1==1):#全局变量
            MSR_AS=0#0时表示不分库，1分库
            TRAIN_RATIO=2/3#MSR_AS=1时有用
            
            CROSS_SUBJECT=1#1表示主题交叉验证
        
            PCA_NUM1=64
            PCA_NUM2=860
            LOOP_NUM=10#子空间投影，投影矩阵迭代次数
            
            
        if(1==1):#全局设置
            if(MSR_AS==0):
                CLASS_NUM=27
                ALL_SAMPLE_NUM=861
                TRAIN_NUM=int(ALL_SAMPLE_NUM*TRAIN_RATIO)#训练样本数
                TEST_NUM = ALL_SAMPLE_NUM - TRAIN_NUM  # 测试样本数
                print('###########UTD全部样本############')
    if(1==1):#UTDV2
        if(1==1):#全局变量
            MSR_AS=0#0时表示不分库，1分库
            TRAIN_RATIO=1/2#MSR_AS=1时有用
            
            CROSS_SUBJECT=1#1表示主题交叉验证
        
            PCA_NUM1=300
            PCA_NUM2=300
            LOOP_NUM=10#子空间投影，投影矩阵迭代次数
            
            
        if(1==1):#全局设置
            if(MSR_AS==0):
                CLASS_NUM=10
                ALL_SAMPLE_NUM=300
                TRAIN_NUM=int(ALL_SAMPLE_NUM*TRAIN_RATIO)#训练样本数
                TEST_NUM = ALL_SAMPLE_NUM - TRAIN_NUM  # 测试样本数
                print('###########UTD全部样本############')            
    if(1==0):#DHA
        if (1 == 1):  # 全局变量
            MSR_AS = 0  # 0时表示不分库，1分库
            TRAIN_RATIO = 2 / 3  # MSR_AS=1时有用

            CROSS_SUBJECT = 1  # 1表示主题交叉验证

            PCA_NUM1 = 483
            PCA_NUM2 = 483
            LOOP_NUM = 10  # 子空间投影，投影矩阵迭代次数

        if (1 == 1):  # 全局设置
            if (MSR_AS == 0):
                CLASS_NUM = 23
                ALL_SAMPLE_NUM = 483
                TRAIN_NUM = int(ALL_SAMPLE_NUM * TRAIN_RATIO)  # 训练样本数
                TEST_NUM = ALL_SAMPLE_NUM - TRAIN_NUM  # 测试样本数
                print('###########DHA全部样本############')
if(1==1):#相关调用函数
    if(1==1):#工具函数
        def DeepCopy(Objects):#深拷贝

           objects=copy.deepcopy(Objects)

           return objects
        def sigmoid(self, x):
            This_x=DeepCopy(x)
            return 1.0 / (1 + np.exp(-This_x))   
        def Confusion_Matrix_a(Listss):#混淆矩阵特征，每行识别率，总识别率。
           listss=DeepCopy(Listss)
           sum_s=0
           sum_d=0
           l=[]
           hh=[]
           for i in range(len(listss)):
              h=[]
              h.append(i)
              h.append(listss[i][i]/sum(listss[i]))
              hh.append(h)
              sum_s=sum_s+sum(listss[i])
              sum_d=sum_d+listss[i][i]
           l.append(sum_d)
           l.append(sum_s)
           l.append(sum_d/sum_s)  
           return (hh,l)
        def Transform_Y(Array_y):#类别标签与01标签转换
                array_y=DeepCopy(Array_y)
                if(isinstance(array_y[0], np.ndarray)):
                    temp_y=[]
                    for i in range(len(array_y)):
                        temp_y.append(np.argmax(array_y[i])+1)
                        # for j in range(CLASS_NUM):
                            # if(array_y[i][j]==1):
                                # temp_y.append(j+1)
                    return array(temp_y)
                    #print(type(array_y[0]))
                else:
                    temp_y=zeros((len(array_y),CLASS_NUM))
                    for i in range(len(array_y)):
                        temp_y[i][array_y[i]-1]=1
                    return temp_y
        def Sample_class_size(train_X,train_Y,Class_num):
            #返回一个1*类别数 的以为数组，统计每一类样本的数量
            this_train_X=DeepCopy(train_X)
            this_train_Y=DeepCopy(train_Y)#123标签
            this_Class_num=DeepCopy(Class_num)
            # print('vvv')
            the_num_of_each_class=zeros((1,this_Class_num))
            for i in range(shape(train_X)[0]):
                the_num_of_each_class[0][this_train_Y[i]-1]=the_num_of_each_class[0][this_train_Y[i]-1]+1#每类样本数量
            return the_num_of_each_class
        def Sample_Mean(train_X,train_Y,Class_num):
            this_train_X=DeepCopy(train_X)
            this_train_Y=DeepCopy(train_Y)#123标签
            this_Class_num=DeepCopy(Class_num)
            # print('vvv')
            sample_mean=zeros((this_Class_num,shape(train_X)[1]))
            the_num_of_each_class=zeros((1,this_Class_num))
            for i in range(shape(train_X)[0]):
                the_num_of_each_class[0][this_train_Y[i]-1]=the_num_of_each_class[0][this_train_Y[i]-1]+1#每类样本数量
                sample_mean[this_train_Y[i]-1]=sample_mean[this_train_Y[i]-1]+train_X[i]#每类样本和
            # print(the_num_of_each_class)
            # print(sample_mean)
            sample_mean_Y=[]
            for i in range(this_Class_num):
                sample_mean[i]=sample_mean[i]/the_num_of_each_class[0][i]
                sample_mean_Y.append(i+1)
            # print(sample_mean)    
            return (sample_mean,sample_mean_Y)
            
        def Sample_Mean_Replace_Sample(train_X,train_Y,Class_num):
            this_train_X=DeepCopy(train_X)
            this_train_Y=DeepCopy(train_Y)
            this_Class_num=DeepCopy(Class_num)#Class_num=20
            
            sample_mean=zeros((this_Class_num,shape(train_X)[1]))
            the_num_of_each_class=zeros((1,this_Class_num))
            for i in range(shape(train_X)[0]):
                the_num_of_each_class[0][this_train_Y[i]-1]=the_num_of_each_class[0][this_train_Y[i]-1]+1#每类样本数量
                sample_mean[this_train_Y[i]-1]=sample_mean[this_train_Y[i]-1]+train_X[i]#每类样本和
            #print(the_num_of_each_class)
            #print(sample_mean)
            #sample_mean_Y=[]
            for i in range(this_Class_num):
                sample_mean[i]=sample_mean[i]/the_num_of_each_class[0][i]
                #sample_mean_Y.append(i+1)
            sample_mean_replace_sample=[]
            for i in range(shape(train_X)[0]):
                sample_mean_replace_sample.append(sample_mean[this_train_Y[i]-1])
            # print(the_num_of_each_class[0])    
            return (sample_mean_replace_sample,this_train_Y)
        def HCC_Replace_Sample(train_X,train_Y,Class_num):
            this_train_X=array(DeepCopy(train_X))
            this_train_Y=array(DeepCopy(train_Y))
            this_Class_num=DeepCopy(Class_num)#Class_num=20
            
            sample_mean=zeros((this_Class_num,shape(this_train_X)[1]))
            the_num_of_each_class=zeros((1,this_Class_num))
            for i in range(shape(this_train_X)[0]):
                the_num_of_each_class[0][this_train_Y[i]-1]=the_num_of_each_class[0][this_train_Y[i]-1]+1#每类样本数量
                # sample_mean[this_train_Y[i]-1]=sample_mean[this_train_Y[i]-1]+this_train_X[i]#每类样本和
            index_start=0
            index_end=the_num_of_each_class[0][0]
            for i in range(this_Class_num):
                # print(index_start)
                # print(index_end)
                index_start=int(index_start)
                index_end=int(index_end)
                
                class_sample_matrix=this_train_X[index_start:index_end,:]
                # print(class_sample_matrix)
                for j in range(shape(this_train_X)[1]):#对每一列进行操作
                    jth_FV=class_sample_matrix[:,j:j+1]
                    jth_FV=jth_FV.reshape(1,int(the_num_of_each_class[0][i]))[0]
                    jth_FV_sort_minTomax=jth_FV.argsort()
                    # jth_FV_sort_maxTomin=(-jth_FV).argsort()
                    for m in range(int((int(the_num_of_each_class[0][i]+1))/2)):#
                        if(m%2==0):
                            jth_FV[jth_FV_sort_minTomax[m]]=0
                        else:
                            jth_FV[jth_FV_sort_minTomax[-int((m+1)/2)]]=0
                        # jth_FV[jth_FV_sort_maxTomin[m]]=0
                    # print('111111111111')
                    # jth_FV[-1]=1
                    # print(jth_FV)
                    # print(jth_FV_sort_minTomax)
                    sum(jth_FV)
                    sample_mean[i][j]=sum(jth_FV)/(int(the_num_of_each_class[0][i])-int((the_num_of_each_class[0][i]+1)/2))
                    # print(jth_FV_sort_maxTomin)
                if(i<this_Class_num-1):#最后一个不计算下一次循环index
                    index_start=index_start+the_num_of_each_class[0][i]
                    index_end=index_end+the_num_of_each_class[0][i+1]
                

            # for i in range(this_Class_num):
                # sample_mean[i]=sample_mean[i]/the_num_of_each_class[0][i]
            HCC_replace_sample=[]
            for i in range(shape(this_train_X)[0]):
                HCC_replace_sample.append(list(sample_mean[this_train_Y[i]-1]))
            return (HCC_replace_sample,this_train_Y)
        def SVD(dataMat, topNfeat=9999999):
            this_dataMat=DeepCopy(dataMat)
            
            U,Sigma,VT=linalg.svd(this_dataMat)
            print("Sigma\n",Sigma)
            return np.dot(this_dataMat, VT.T[:,0:topNfeat])
        def MSR_Divide(X,Y,Num):
            this_x=DeepCopy(X)
            this_y=DeepCopy(Y)
            
            x_as1=[]
            y_as1=[]
            x_as2=[]
            y_as2=[]
            x_as3=[]
            y_as3=[]
            for i in range(shape(x)[0]):
                #AS1
                if(this_y[i][0]==2 or this_y[i][0]==3 or this_y[i][0]==5 or this_y[i][0]==6 or this_y[i][0]==10 or this_y[i][0]==13 or this_y[i][0]==18 or this_y[i][0]==20):
                    x_as1.append(x[i])
                    y_as1.append(this_y[i])
                #AS2
                if(this_y[i][0]==1 or this_y[i][0]==4 or this_y[i][0]==7 or this_y[i][0]==8 or this_y[i][0]==9 or this_y[i][0]==11 or this_y[i][0]==12 or this_y[i][0]==14):
                    x_as2.append(x[i])
                    y_as2.append(this_y[i])
                #AS3
                if(this_y[i][0]==6 or this_y[i][0]==14 or this_y[i][0]==15 or this_y[i][0]==16 or this_y[i][0]==17 or this_y[i][0]==18 or this_y[i][0]==19 or this_y[i][0]==20):
                    x_as3.append(x[i])
                    y_as3.append(this_y[i])
            if(Num==1):
                this_x=array(x_as1)
                this_y=array(y_as1)

            if(Num==2):
                this_x=array(x_as2)
                this_y=array(y_as2)

            if(Num==3):
                this_x=array(x_as3)
                this_y=array(y_as3)
            return (this_x,this_y)
        def CrossSubject(X, Y_Label):
            this_x = DeepCopy(X)
            this_y_label = DeepCopy(Y_Label)

            x_tr = []
            y_tr = []
            x_te = []
            y_te = []
            for i in range(shape(this_x)[0]):
                if (this_y_label[i][1]%2==1):
                    x_tr.append(this_x[i])
                    y_tr.append(this_y_label[i][0])
                if (this_y_label[i][1]%2==0):
                    x_te.append(this_x[i])
                    y_te.append(this_y_label[i][0])
            return (array(x_tr), array(x_te), array(y_tr), array(y_te))
        def MSR_CrossSubject(X,Y_Label):
            this_x=DeepCopy(X)
            this_y_label=DeepCopy(Y_Label)
            
            x_tr=[]
            y_tr=[]
            x_te=[]
            y_te=[]
            for i in range(shape(x)[0]):
                if(this_y_label[i][1]==1 or this_y_label[i][1]==3 or this_y_label[i][1]==5 or this_y_label[i][1]==7 or this_y_label[i][1]==9):
                    #print(this_y_label[i])
                    x_tr.append(x[i])
                    y_tr.append(this_y_label[i][0])
                if(this_y_label[i][1]==2 or this_y_label[i][1]==4 or this_y_label[i][1]==6 or this_y_label[i][1]==8 or this_y_label[i][1]==10):
                    x_te.append(x[i])
                    y_te.append(this_y_label[i][0])
            return (array(x_tr),array(x_te),array(y_tr),array(y_te))
        def Center_Array(Array):
            This_array=array(DeepCopy(Array))
            
            meanVals = mean(This_array, axis=0)     #按列求均值，即每一列求一个均值，不同的列代表不同的特征               
            meanRemoved = This_array - meanVals
            return meanRemoved
    if(1==1):#神经网络相关函数
        #implement the neural network
        def sigmoid(x):
            return 1/(1+np.exp(-x))
         
        def dsigmoid(x):
            return x*(1-x)

        class NeuralNetwork(object):
            def __init__(self,input_size,hidden_size,output_size):
                self.W1 = 0.01 * np.random.randn(input_size,hidden_size)#D*H
                self.b1 = np.zeros(hidden_size) #H
                self.W2 = 0.01 * np.random.randn(hidden_size,output_size)#H*C
                self.b2 = np.zeros(output_size)#C
            
            def loss(self,X,y,reg = 0.01):
                num_train, num_feature = X.shape
                #forward
                a1 = X  #input layer:N*D
                a2 = sigmoid(a1.dot(self.W1) + self.b1) #hidden layer:N*H
                a3 = sigmoid(a2.dot(self.W2) + self.b2) #output layer:N*C
                
                loss = - np.sum(y*np.log(a3) + (1-y)*np.log((1-a3)))/num_train
                loss += 0.5 * reg * (np.sum(self.W1*self.W1)+np.sum(self.W2*self.W2)) / num_train
                
                #backward
                error3 = a3 - y #N*C
                dW2 = a2.T.dot(error3) + reg * self.W2#(H*N)*(N*C)=H*C
                db2 = np.sum(error3,axis=0)
                
                error2 = error3.dot(self.W2.T)*dsigmoid(a2) #N*H
                dW1 = a1.T.dot(error2) + reg * self.W1     #(D*N)*(N*H) =D*H
                db1 = np.sum(error2,axis=0)
                
                dW1 /= num_train
                dW2 /= num_train
                db1 /= num_train
                db2 /= num_train
                
                return loss,dW1,dW2,db1,db2
            
            def train(self,X,y,y_train,X_val,y_val,learn_rate=0.01,num_iters = 5000):
                batch_size = 150
                num_train = X.shape[0]
                loss_list = []
                accuracy_train = []
                accuracy_val = []
                
                for i in range(num_iters):
                    batch_index = np.random.choice(num_train,batch_size,replace=True)
                    X_batch = X[batch_index]
                    y_batch = y[batch_index]
                    y_train_batch = y_train[batch_index]
                    
                    loss,dW1,dW2,db1,db2 = self.loss(X_batch,y_batch)
                    loss_list.append(loss)
                    
                    #update the weight
                    self.W1 += -learn_rate*dW1
                    self.W2 += -learn_rate*dW2
                    self.b1 += -learn_rate*db1
                    self.b2 += -learn_rate*db2
                    
                    if i%500 == 0:
                        print ("i=%d,loss=%f" %(i,loss))
                        #record the train accuracy and validation accuracy
                        train_acc = np.mean(y_train_batch==self.predict(X_batch))
                        val_acc = np.mean(y_val==self.predict(X_val))
                        accuracy_train.append(train_acc)
                        accuracy_val.append(val_acc)
                        
                return loss_list,accuracy_train,accuracy_val
            
            def predict(self,X_test):
                a2 = sigmoid(X_test.dot(self.W1) + self.b1)
                a3 = sigmoid(a2.dot(self.W2) + self.b2)
                y_pred = np.argmax(a3,axis=1)
                return y_pred
                
                
            pass
    if(1==1):#极限学习机相关函数
        class SingeHiddenLayer(object):

            def __init__(self, X, y, num_hidden):
                self.data_x = X  # 判断输入训练集是否大于等于二维; 把x_train()取下来
                self.data_y = y # a.flatten()把a放在一维数组中，不写参数默认是“C”，也就是先行后列的方式，也有“F”先列后行的方式； 把 y_train取下来
                self.num_data = shape(X)[0]  # 训练数据个数
                self.num_feature = self.data_x.shape[1];  # shape[] 读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度 (120行，4列，所以shape[0]==120,shapep[1]==4)
                self.num_hidden = num_hidden;  # 隐藏层节点个数

                # 随机生产权重（从-1，到1，生成（num_feature行,num_hidden列））
                self.w = np.random.uniform(-1, 1, (self.num_feature, self.num_hidden))#特征数*隐层数(投影数)
                # print(shape(self.w))

                # 随机生成偏置，一个隐藏层节点对应一个偏置
                for i in range(self.num_hidden):#
                    b = np.random.uniform(-0.6, 0.6, (1, self.num_hidden))
                    self.first_b = b
                    # print(shape(b))
                    # print(b)
                # print(shape(b))
                # print(b)    
                # print(shape(self.first_b))
                # print(self.first_b)
                # 生成偏置矩阵，以隐藏层节点个数4为行，样本数120为列
                for i in range(self.num_data - 1):
                    b = np.row_stack((b, self.first_b))  # row_stack 以叠加行的方式填充数组
                self.b = b
                # print(shape(self.b))
                # print(b)
                # print(shape(self.b))
                # print(self.b)
            # 定义sigmoid函数
            def sigmoid(self, x):
                return 1.0 / (1 + np.exp(-x))

            def train(self, x_train, y_train, classes):
                # print('x:'+str(shape(x_train)))
                # print('y:'+str(shape(y_train)))
                mul = np.dot(self.data_x, self.w)  # 输入乘以权重
                # print(shape(mul))
                # print('b:'+str(shape(self.b )))
                # print('mul:'+str(shape(mul )))
                add = mul + self.b  # 加偏置
                # print(shape(add ))
                H = self.sigmoid(add)  # 激活函数
                
                # print('H:'+str(shape(H)))    
                H_ = np.linalg.pinv(H)  # 求广义逆矩阵
                # print(shape(H_))
                # print(type(H_.shape))

                # 将只有一列的Label矩阵转换，例如，iris的label中共有三个值，则转换为3列，以行为单位，label值对应位置标记为1，其它位置标记为0
                self.train_y = np.zeros((self.num_data, classes))  # 初始化一个120行，3列的全0矩阵
                for i in range(0, self.num_data):
                    self.train_y[i, y_train[i]] = 1  # 对应位置标记为1
                # print(shape(self.train_y))
                self.out_w = np.dot(H_, self.train_y)  # 求输出权重

            def predict(self, x_test):
                self.t_data = np.atleast_2d(x_test)  # 测试数据集
                self.num_tdata = len(self.t_data)  # 测试集的样本数
                self.pred_Y = np.zeros((x_test.shape[0]))  # 初始化

                b = self.first_b

                # 扩充偏置矩阵，以隐藏层节点个数4为行，样本数30为列
                for i in range(self.num_tdata - 1):
                    b = np.row_stack((b, self.first_b))  # 以叠加行的方式填充数组

                # 预测
                self.pred_Y = np.dot(self.sigmoid(np.dot(self.t_data, self.w) + b), self.out_w)

                # 取输出节点中值最大的类别作为预测值
                self.predy = []
                for i in self.pred_Y:
                    L = i.tolist()
                    self.predy.append(L.index(max(L)))

            def Confusion_Matrix_a(self, Listss):  # 混淆矩阵特征，每行识别率，总识别率。
                listss = DeepCopy(Listss)
                sum_s = 0
                sum_d = 0
                l = []
                hh = []
                for i in range(len(listss)):
                    h = []
                    h.append(i)
                    h.append(listss[i][i] / sum(listss[i]))
                    hh.append(h)
                    sum_s = sum_s + sum(listss[i])
                    sum_d = sum_d + listss[i][i]
                l.append(sum_d)
                l.append(sum_s)
                l.append(sum_d / sum_s)
                return (hh, l)

            def score(self, y_test):
                print("准确率：")
                # 使用准确率方法验证
                mat = confusion_matrix(y_test, self.predy)  # 生成每折混淆矩阵
                # sio.savemat(PATH_R1 + 'ELM.mat')
                print(mat)
                (a, b) = self.Confusion_Matrix_a(mat)
                print(np.array(a))  # 每种行为识别率
                print(b)  # 总体识别率
                # sio.savemat(PATH_R1 + 'ELM.mat')
            pass        
    if(1==1):#子空间投影相关函数
        if(1==1):#步骤1(针对每类样本不按顺序，或者相同类别样本不近邻)#行为识别样本可直接使用转置与编码转化
            def build_XT_and_Y_simple(train_X,train_Y):##单模态
                #单模态
                
                #1、X: X=x.T
                #2、Y: 训练样本的子空间

                this_x=DeepCopy(train_X)
                this_y=DeepCopy(train_Y)
                
                return (XT,Y)
            def build_XT_and_Y(train_X,train_Y):##单模态
                #单模态
                
                #1、X: X=x.T
                #2、Y: 训练样本的子空间

                this_x=DeepCopy(train_X)
                this_y=DeepCopy(train_Y)
                
                train_num, feature_num=shape(this_x)
                
                ###############################
                #1、使用class_num记录训练样本里各个类个数
                #2、将训练样本this_x放到（类，各类训练样本数，特征数）3维矩阵class_samples_image中
                if(1==1):
                    #1、使用class_num记录训练样本里各个类个数
                    #2、将训练样本放到（类，各类训练样本数，特征数）3维矩阵class_samples_image中
                
                    class_num = zeros((1, CLASS_NUM))#记录训练数据里各个类个数
                    class_samples= zeros((CLASS_NUM, train_num, feature_num))#（类，训练样本数，特征数）
                    for num in range(train_num):#将训练样本放到（类，训练样本数，特征数）3维矩阵中
                        for c_n in range(CLASS_NUM):
                            if(this_y[num]==c_n+1):#类序号从1开始
                                class_samples[c_n][int(class_num[0][c_n])][:]= this_x[num][:]
                                
                                class_num[0][c_n] = class_num[0][c_n] + 1
                
                ########################
                #1、Y:制作训练样本目标投影空间Y（训练样本数，类别数）
                #2、X:将训练样本this_x（训练样本数，特征数）转置，变成X（特征数，训练样本数）
                if(1==1):
                    #1、制作训练样本目标投影空间Y（训练样本数，类别数）
                    #2、将训练样本this_x（训练样本数，特征数）转置，变成X（特征数，训练样本数）
                    # %就是将图片文本特征按顺序放到图片文本容器转置
                    XT = zeros((feature_num, train_num))#容器转置X（特征数，训练样本数）
                    Y = zeros((train_num, CLASS_NUM))#投影空间初始化（样本数*类别数）全0
                    
                    Y[0:int(class_num[0][0]),0:1] = 1#
                    temp= zeros((int(class_num[0][0]), feature_num))#(第一类样本数，特征数)
                    temp[0:int(class_num[0][0]), :] = class_samples[0][0:int(class_num[0][0]),:] #(每一类样本数，特征数)=(类，训练样本数，特征数)[0]
                    XT[0:feature_num, 0:int(class_num[0][0])] = array(mat(temp).T)#第一类放到图片容器转置
                    for j in range(1,CLASS_NUM):
                        tsum=0
                        for t in range(j):#计算j类之前的样本数
                            tsum=tsum+class_num[0][t]
                        Y[int(tsum):int(tsum+class_num[0][j]),j]=1#
                        
                        temp = zeros((int(class_num[0][j]), feature_num)) 
                        temp[0:int(class_num[0][j]), :] = class_samples[j][0:int(class_num[0][j]),:]#(每一类样本数，特征数)=(类，训练样本数，特征数)[0]
                        XT[0:feature_num, int(tsum):int(tsum+class_num[0][j])] = array(mat(temp).T)#第一类放到图片容器转置
                ########################

                return (XT,Y)
            def build_XT_and_Y_and_YY(train_X,train_Y):##单模态多聚点
                #单模态
                
                #1、X: X=x.T
                #2、Y: 训练样本的子空间
                #3、YY：多聚点标签

                this_x=DeepCopy(train_X)
                this_y=DeepCopy(train_Y)
                
                train_num, feature_num=shape(this_x)
                
                ###############################
                #1、使用class_num记录训练样本里各个类个数
                #2、将训练样本this_x放到（类，各类训练样本数，特征数）3维矩阵class_samples_image中
                if(1==1):
                    #1、使用class_num记录训练样本里各个类个数
                    #2、将训练样本放到（类，各类训练样本数，特征数）3维矩阵class_samples_image中
                
                    class_num = zeros((1, CLASS_NUM))#记录训练数据里各个类个数
                    class_samples= zeros((CLASS_NUM, train_num, feature_num))#（类，训练样本数，特征数）
                    for num in range(train_num):#将训练样本放到（类，训练样本数，特征数）3维矩阵中
                        for c_n in range(CLASS_NUM):
                            if(this_y[num]==c_n+1):#类序号从1开始
                                class_samples[c_n][int(class_num[0][c_n])][:]= this_x[num][:]
                                
                                class_num[0][c_n] = class_num[0][c_n] + 1
                
                ########################
                #1、Y:制作训练样本目标投影空间Y（训练样本数，类别数）
                #2、X:将训练样本this_x（训练样本数，特征数）转置，变成X（特征数，训练样本数）
                if(1==1):
                    #1、制作训练样本目标投影空间Y（训练样本数，类别数）
                    #2、将训练样本this_x（训练样本数，特征数）转置，变成X（特征数，训练样本数）
                    # %就是将图片文本特征按顺序放到图片文本容器转置
                    XT = zeros((feature_num, train_num))#容器转置X（特征数，训练样本数）
                    Y = zeros((train_num, CLASS_NUM))#投影空间初始化（样本数*类别数）全0
                    
                    Y[0:int(class_num[0][0]),0:1] = 1#
                    temp= zeros((int(class_num[0][0]), feature_num))#(第一类样本数，特征数)
                    temp[0:int(class_num[0][0]), :] = class_samples[0][0:int(class_num[0][0]),:] #(每一类样本数，特征数)=(类，训练样本数，特征数)[0]
                    XT[0:feature_num, 0:int(class_num[0][0])] = array(mat(temp).T)#第一类放到图片容器转置
                    for j in range(1,CLASS_NUM):
                        tsum=0
                        for t in range(j):#计算j类之前的样本数
                            tsum=tsum+class_num[0][t]
                        Y[int(tsum):int(tsum+class_num[0][j]),j]=1#
                        
                        temp = zeros((int(class_num[0][j]), feature_num)) 
                        temp[0:int(class_num[0][j]), :] = class_samples[j][0:int(class_num[0][j]),:]#(每一类样本数，特征数)=(类，训练样本数，特征数)[0]
                        XT[0:feature_num, int(tsum):int(tsum+class_num[0][j])] = array(mat(temp).T)#第一类放到图片容器转置
                ########################
                #1、YY:制作训练样本目标投影空间Y（训练样本数，类别数）
                if(1==1):
                    YY_list=[]
                    temp_y_a = zeros((shape(train_X)[0],CLASS_NUM))
                    temp_y_a[0:shape(train_X)[0],:]=Y[0:shape(train_X)[0],:]
                    for i in range(CLASS_NUM-1):

                        temp_y_b = zeros((shape(train_X)[0], CLASS_NUM))
                        for j in range(CLASS_NUM):
                            if(j==0):
                                temp_y_b[:,j]=temp_y_a[:,CLASS_NUM-1]
                            else:
                                temp_y_b[:,j]=temp_y_a[:,j-1]
                        temp_y_a[0:shape(train_X)[0],:]=temp_y_b[0:shape(train_X)[0],:]
                        YY_list.append(Y*2-temp_y_a)
                        
                    
                return (XT,Y,YY_list)
            def build_X_and_Y_and_W_and_D_and_L(train_X,train_Y):##单模态
                #单模态
                #1、XT: X=x.T
                #2、Y: 训练样本的子空间
                #3、W: 模态内相似度矩阵
                #4、D: 模态内度矩阵
                #5、L: 模态内拉普拉斯矩阵
                this_x=DeepCopy(train_X)
                this_y=DeepCopy(train_Y)
                
                train_num, feature_num=shape(this_x)
                
                ###############################
                #1、使用class_num记录训练样本里各个类个数
                #2、将训练样本this_x放到（类，各类训练样本数，特征数）3维矩阵class_samples_image中
                if(1==1):
                    #1、使用class_num记录训练样本里各个类个数
                    #2、将训练样本放到（类，各类训练样本数，特征数）3维矩阵class_samples_image中
                
                    class_num = zeros((1, CLASS_NUM))#记录训练数据里各个类个数
                    class_samples= zeros((CLASS_NUM, train_num, feature_num))#（类，训练样本数，特征数）
                    for num in range(train_num):#将训练样本放到（类，训练样本数，特征数）3维矩阵中
                        for c_n in range(CLASS_NUM):
                            if(this_y[num]==c_n+1):
                                class_samples[c_n][int(class_num[0][c_n])][:]= this_x[num][:]
                                
                                class_num[0][c_n] = class_num[0][c_n] + 1
                
                ########################
                #1、Y:制作训练样本目标投影空间Y（训练样本数，类别数）
                #2、X:将训练样本this_x（训练样本数，特征数）转置，变成X（特征数，训练样本数）
                if(1==1):
                    #1、制作训练样本目标投影空间Y（训练样本数，类别数）
                    #2、将训练样本this_x（训练样本数，特征数）转置，变成X（特征数，训练样本数）
                    # %就是将图片文本特征按顺序放到图片文本容器转置
                    XT = zeros((feature_num, train_num))#容器转置X（特征数，训练样本数）
                    Y = zeros((train_num, CLASS_NUM))#投影空间初始化（样本数*类别数）全0
                    
                    Y[0:int(class_num[0][0]),0:1] = 1#
                    temp= zeros((int(class_num[0][0]), feature_num))#(每一类样本数，特征数)
                    temp[0:int(class_num[0][0]), :] = class_samples[0][0:int(class_num[0][0]),:] #(每一类样本数，特征数)=(类，训练样本数，特征数)[0]
                    XT[0:feature_num, 0:int(class_num[0][0])] = array(mat(temp).T)#第一类放到图片容器转置
                    for j in range(1,CLASS_NUM):
                        tsum=0
                        for t in range(j):#计算j类之前的样本数
                            tsum=tsum+class_num[0][t]
                        Y[int(tsum):int(tsum+class_num[0][j]),j]=1#
                        
                        temp = zeros((int(class_num[0][j]), feature_num)) 
                        temp[0:int(class_num[0][j]), :] = class_samples[j][0:int(class_num[0][j]),:]#(每一类样本数，特征数)=(类，训练样本数，特征数)[0]
                        XT[0:feature_num, int(tsum):int(tsum+class_num[0][j])] = array(mat(temp).T)#第一类放到图片容器转置
                
                ############################
                #1、T:计算各个模态内样本之间距离生成T（训练样本数，训练样本数）
                #2、W:样本之间距离，生成由各个模态内相似度矩阵W
                if(1==1):
                    #1、T:计算各个模态内样本之间距离生成T（训练样本数，训练样本数）
                    #2、W:样本之间距离，生成由各个模态内相似度矩阵W
                    
                    delta = 0.1#可以用不同的价值产生很大的影响力。
                    
                    W = zeros((train_num, train_num))#模态内
                    T = zeros((train_num, train_num))#模态内样本距离值
                    
                    for i in range(0,train_num):#模态内样本间距离矩阵
                        for j in range(0,train_num):
                            T[i][j]=sum((XT[:, i] - XT[:, j])*(XT[:, i] - XT[:, j]))
                    k_num = train_num#近邻数
                    for i in range(train_num):#模态内样本间相似度矩阵
                        for j in range(k_num):#
                            W[i][j]=exp(-T[i][j]*(1/(2*delta*delta)))

                ################################
                #1、D：制作模态内度矩阵D(训练样本数，训练样本数)
                #1、L：制作模态内拉普拉斯矩阵D(训练样本数，训练样本数)
                if(1==1):
                    #1、D：制作模态内度矩阵D(训练样本数，训练样本数)
                    #1、L：制作模态内拉普拉斯矩阵D(训练样本数，训练样本数)
                    beta = 1#影响不大
                    D = zeros((train_num, train_num))#模态内
                    for i in range(0,train_num):
                        for j in range(0,train_num):
                            if(W[i][j] > 0):
                                D[i][i] = D[i][i]+ 1
                    L= D- beta * W 
                
                return (XT,Y,L)
            def build_WW_and_DD_and_LL(train_X_1,train_X_2,train_Y):##双模态
                #双模态
                #3、WW: 模态间相似度矩阵
                #4、DD: 模态间度矩阵
                #5、LL: 模态间拉普拉斯矩阵
                this_x_1=DeepCopy(train_X_1)
                this_x_2=DeepCopy(train_X_2)
                this_y=DeepCopy(train_Y)
                
                train_num, feature_num=shape(this_x_1)
                
                ###############################
                #1、使用class_num记录训练样本里各个类个数
                if(1==1):
                    #1、使用class_num记录训练样本里各个类个数
                    #2、将训练样本放到（类，各类训练样本数，特征数）3维矩阵class_samples_image中
                
                    class_num = zeros((1, CLASS_NUM))#记录训练数据里各个类个数
                    class_samples= zeros((CLASS_NUM, train_num, feature_num))#（类，训练样本数，特征数）
                    for num in range(train_num):#将训练样本放到（类，训练样本数，特征数）3维矩阵中
                        for c_n in range(CLASS_NUM):
                            if(this_y[num]==c_n+1):
                                class_num[0][c_n] = class_num[0][c_n] + 1
                
                ###########
                #1、WW:制作两个模态间相似度矩阵WW
                #2、DD:制作两个模态间度矩阵DD
                #3、LL:制作两个模态间拉普拉斯矩阵LL
                if(1==1):
                    #1、WW:制作两个模态间相似度矩阵WW
                    WW = zeros((train_num, train_num))#模态间
                    DD = zeros((train_num, train_num))
                    WW[0:int(class_num[0][0]), 0:int(class_num[0][0])] = 1#第一类方阵设置为1
                    for j in range(1,CLASS_NUM):
                        tsum=0
                        for t in range(j):#计算j类之前的样本数
                            tsum=tsum+class_num[0][t]
                        WW[int(tsum):int(tsum+class_num[0][j]),int(tsum):int(tsum+class_num[0][j])] = 1
                        
                    for i in range(train_num):#2、DD:制作两个模态间度矩阵DD
                        DD[i][i] = sum(WW[i])#对角方阵到对角线
                    
                    LL = DD - WW#3、LL:制作两个模态间拉普拉斯矩阵LL
                return (LL)
        if(1==1):#步骤2
            def build_R_SM(U,XT):
                this_U=DeepCopy(U)
                this_XT=DeepCopy(XT)

                feature_num,train_num =shape(this_XT)
                
                small_value = 1
                
                R= zeros((feature_num, feature_num))
                temp= (this_U**2).sum(1)
                for num in range(feature_num):
                    R[num][num] =1/(2*sqrt(temp[num]+small_value))
                    
                return R
        if(1==1):#步骤3
            def update_U_ELM_SM(U,XT,Y,lambda_List):##极限学习机单模态
                This_U=DeepCopy(U)
                This_XT=DeepCopy(XT)
                This_Y=DeepCopy(Y)

           
                
                #########
                #1、R
                # This_R=build_R_SM(This_U,This_XT)
                
                #lambda1 =200#0.00000000005-1
                lambda1=lambda_List[0]
                if(1==1):#单模态
                    This_XT=mat(This_XT)
                    # This_R=mat(This_R)
                    This_Y=mat(This_Y)
                    This_XT_ = np.linalg.pinv(This_XT.T)
                    This_U=array(This_XT_*This_Y)
                    # This_U=array(((This_XT* This_XT.T).I)
                    # * (This_XT*This_Y))

                return This_U
            def update_U_SL_SM(U,XT,Y,lambda_List):##单模态
                This_U=DeepCopy(U)
                This_XT=DeepCopy(XT)
                This_Y=DeepCopy(Y)

           
                
                #########
                #1、R
                This_R=build_R_SM(This_U,This_XT)
                
                #lambda1 =200#0.00000000005-1
                lambda1=lambda_List[0]
                if(1==1):#单模态
                    This_XT=mat(This_XT)
                    This_R=mat(This_R)
                    This_Y=mat(This_Y)
                    This_U=array(((This_XT* This_XT.T+lambda1* This_R).I)
                    * (This_XT*This_Y))
                    

                return This_U
            def update_U_MCSL_SM(U,XT,Y,YY_list,lambda_List):##单模态多聚点
                This_U=DeepCopy(U)
                This_XT=DeepCopy(XT)
                This_Y=DeepCopy(Y)
                This_YY_list=DeepCopy(YY_list)

           
                Class_num=shape(This_Y)[1]
                #########
                #1、R
                This_R=build_R_SM(This_U,This_XT)
                
                #lambda1 = 200
                lambda1=lambda_List[0]
                if(1==1):#单模态
                    This_XT=mat(This_XT)
                    This_R=mat(This_R)
                    This_Y=mat(This_Y)
                    YY_NUM=mat(zeros((len(This_XT),CLASS_NUM)))
                    for i in range(CLASS_NUM-1):
                        YY_NUM=YY_NUM+mat(This_XT)*mat(This_YY_list[i])
                    This_U=array(((Class_num*This_XT* This_XT.T+lambda1* This_R).I)
                    * (This_XT*This_Y+YY_NUM))

                return This_U
            def update_U_DDSL_SM(U,XT,Y,HCC_replace_sample_T,lambda_List):
                This_U=DeepCopy(U)
                This_XT=DeepCopy(XT)
                This_Y=DeepCopy(Y)
                This_HCC_replace_sample_T=DeepCopy(HCC_replace_sample_T)    
                
                #########
                #1、R
                This_R=build_R_SM(This_U,This_XT)
                
                # lambda1 = 0.2
                # lambda2 = 1
                lambda0=1
                lambda1=lambda_List[0]
                lambda2=lambda_List[1]
                if(1==1):#单模态
                    This_XT=mat(This_XT)
                    This_R=mat(This_R)
                    This_Y=mat(This_Y)
                    This_HCC_replace_sample_T=mat(This_HCC_replace_sample_T)
                    This_U=array(((lambda0*This_XT* This_XT.T+lambda1* This_R+lambda2* This_HCC_replace_sample_T*This_HCC_replace_sample_T.T).I)
                    * (lambda0*This_XT*This_Y+lambda2* This_HCC_replace_sample_T*This_Y))

                return This_U
            def update_U_JFSSL_SM(U,XT,Y,L,lambda_List):
                This_U=DeepCopy(U)
                This_XT=DeepCopy(XT)
                This_Y=DeepCopy(Y)
                This_L=DeepCopy(L)
                
                
                #########
                #1、R_list
                This_R=build_R_SM(This_U,This_XT)
                
                # lambda1 = 0.30
                # lambda2 = 0.1#0.0000001
                
                lambda1=lambda_List[0]
                lambda2=lambda_List[1]
                
                if(1==1):#单模态
                    This_XT=mat(This_XT)
                    This_R=mat(This_R)
                    This_Y=mat(This_Y)
                    This_L=mat(This_L)
                    This_U=array(((This_XT* This_XT.T+lambda1* This_R+lambda2* This_XT*This_L* This_XT.T).I)
                    * (This_XT*This_Y))
                return This_U
            def update_U_JFSSL_MM(U_List,XT_List,Y,L_List,LL,lambda_List):
                U_list=DeepCopy(U_List)
                XT_list=DeepCopy(XT_List)
                this_Y=DeepCopy(Y)
                L_list=DeepCopy(L_List)
                this_LL=DeepCopy(LL)
                
                
                model_num=len(XT_list)#模态数
                
                #########
                #1、R_list
                R_list=[]
                for i in range(model_num):
                    R_list.append( build_R_SM(U_list[i],XT_list[i]))
                
                # lambda1 = 0.30
                # lambda2 = 0.02#0.0000001
                
                lambda1=lambda_List[0]
                lambda2=lambda_List[1]
                
                if(model_num==1):#单模态
                    XT=mat(XT_list[0])
                    R=mat(R_list[0])
                    L=mat(L_list[0])
                    this_Y=mat(this_Y)
                    U_list[0]=array(((XT* XT.T+lambda1* R+lambda2* XT*L* XT.T).I)
                    * (XT*this_Y))
                else:#多模态
                    U_list_last=DeepCopy(U_list)
                    for i in range(model_num):
                        LL_NUM=mat(zeros((len(XT_list[i]),CLASS_NUM)))
                        for j in range(model_num):
                            if(j!=i):
                                LL_NUM=LL_NUM+mat(XT_list[i])*mat(LL)*mat(XT_list[j]).T*mat(U_list_last[j])
                                
                        U_list[i]=array(((mat(XT_list[i]) * mat(XT_list[i]).T+lambda1*mat(R_list[i])+lambda2*mat(XT_list[i])*mat(L_list[i])*mat(XT_list[i]).T).I)
                        *(mat(XT_list[i])*Y-lambda2*LL_NUM))
                        #U_list[i]=array(((mat(X_list[i]) * mat(X_list[i]).T+lambda1*mat(R_list[i])).I)
                        #*(mat(X_list[i])*Y))
                    
                return U_list
        if(1==1):#各类子空间学习
            def SL_SM(train_X,train_Y,loop_num,lambda_List):
                (This_XT,This_Y)=build_XT_and_Y(train_X,train_Y)
                U = eye(shape(This_XT)[0], shape(This_Y)[1])#初始化投影矩阵
                for i in range(loop_num):
                    U=update_U_SL_SM(U,This_XT,This_Y,lambda_List) 
                return U
            def MCSL_SM(train_X,train_Y,loop_num,lambda_List):
                (This_XT,This_Y,This_YY_list)=build_XT_and_Y_and_YY(train_X,train_Y)
                U = eye(shape(This_XT)[0], shape(This_Y)[1])#初始化投影矩阵
                for i in range(loop_num):
                    U=update_U_MCSL_SM(U,This_XT,This_Y,This_YY_list,lambda_List) 
                return U
            def DDSL_SM(train_X,train_Y,loop_num,lambda_List):
                (This_XT,This_Y)=build_XT_and_Y(train_X,train_Y)
                # print(shape(This_XT))#XT：特征数*训练样本数
                # print(shape(This_Y))#Y:训练样本数*类别数
                (HCC_replace_sample,A)=HCC_Replace_Sample(mat(This_XT).T,Transform_Y(This_Y),CLASS_NUM)
                # print(shape(HCC_replace_sample))
                #print(shape(sample_mean_replace_sample))
                HCC_replace_sample_T=mat(HCC_replace_sample).T
                U = eye(shape(This_XT)[0], shape(This_Y)[1])#初始化投影矩阵
                U_list=[]
                for i in range(loop_num): 
                    U=update_U_DDSL_SM(U,This_XT,This_Y,HCC_replace_sample_T,lambda_List) 
                    U_list.append(U)
                return U,U_list
            def XXSL_SM(train_X,train_Y,loop_num,lambda_List):
                (This_XT,This_Y)=build_XT_and_Y(train_X,train_Y)
                # print(shape(This_XT))#XT：特征数*训练样本数
                # print(shape(This_Y))#Y:训练样本数*类别数
                (HCC_replace_sample,A)=HCC_Replace_Sample(mat(This_XT).T,Transform_Y(This_Y),CLASS_NUM)
                # print(shape(HCC_replace_sample))
                #print(shape(sample_mean_replace_sample))
                HCC_replace_sample_T=mat(HCC_replace_sample).T
                U = eye(shape(This_XT)[0], shape(This_Y)[1])#初始化投影矩阵
                U_list=[]
                for i in range(loop_num): 
                    U=update_U_DDSL_SM(U,This_XT,This_Y,HCC_replace_sample_T,lambda_List) 
                    U_list.append(U)
                return U,U_list    
            def FFSL_SM(train_X,train_Y,num_hidden,loop_num,lambda_List):#有HCC
                (This_XT,This_Y)=build_XT_and_Y(train_X,train_Y)
                # print(shape(This_XT))#XT：特征数*训练样本数
                # print(shape(This_Y))#Y:训练样本数*类别数
                (HCC_replace_sample,A)=HCC_Replace_Sample(mat(This_XT).T,Transform_Y(This_Y),CLASS_NUM)
                # print(shape(HCC_replace_sample))
                #print(shape(sample_mean_replace_sample))
                HCC_replace_sample_T=mat(HCC_replace_sample).T
                
                #随机投影矩阵
                w = np.random.uniform(-1, 1, (shape(This_XT)[0],num_hidden))#特征数*隐层数(投影数)
                #偏置向量
                b = np.random.uniform(-0.6, 0.6, (1, num_hidden))
                first_b=b
                #偏置矩阵
                for i in range(shape(This_XT)[1] - 1):
                    b = np.row_stack((b, first_b))  # row_stack 以叠加行的方式填充数组
                x_mul = np.dot(This_XT.T, w)  # 输入乘以权重
                HCC_mul = np.dot(HCC_replace_sample_T.T, w)  # 输入乘以权重
                # print('b:'+str(shape(self.b )))
                # print('mul:'+str(shape(mul )))
                x_add = x_mul + b  # 加偏置
                HCC_add = HCC_mul + b  # 加偏置
                # print(shape(add ))
                new_x = sigmoid(x_add)  # 激活函数
                new_HCC = sigmoid(HCC_add)  # 激活函数
                
                new_x = mat(new_x)
                new_HCC = mat(new_HCC)
                
                U = eye(shape(new_x.T)[0], shape(This_Y)[1])#初始化投影矩阵
                U_list=[]
                for i in range(loop_num): 
                    U=update_U_DDSL_SM(U,new_x.T,This_Y,new_HCC.T,lambda_List) 
                    U_list.append(U)
                return w,first_b,U,U_list  
            def HHSL_SM(train_X,train_Y,num_hidden,loop_num,lambda_List):##极限学习机
                (This_XT,This_Y)=build_XT_and_Y(train_X,train_Y)
                # print(shape(This_XT))#XT：特征数*训练样本数
                # print(shape(This_Y))#Y:训练样本数*类别数
                # (HCC_replace_sample,A)=HCC_Replace_Sample(mat(This_XT).T,Transform_Y(This_Y),CLASS_NUM)
                # print(shape(HCC_replace_sample))
                #print(shape(sample_mean_replace_sample))
                # HCC_replace_sample_T=mat(HCC_replace_sample).T
                
                #随机投影矩阵
                w = np.random.uniform(-1, 1, (shape(This_XT)[0],num_hidden))#特征数*隐层数(投影数)
                #偏置向量
                b = np.random.uniform(-0.6, 0.6, (1, num_hidden))
                first_b=b
                #偏置矩阵
                for i in range(shape(This_XT)[1] - 1):
                    b = np.row_stack((b, first_b))  # row_stack 以叠加行的方式填充数组
                x_mul = np.dot(This_XT.T, w)  # 输入乘以权重
                # HCC_mul = np.dot(HCC_replace_sample_T.T, w)  # 输入乘以权重
                # print('b:'+str(shape(self.b )))
                # print('mul:'+str(shape(mul )))
                x_add = x_mul + b  # 加偏置
                # HCC_add = HCC_mul + b  # 加偏置
                # print(shape(add ))
                new_x = sigmoid(x_add)  # 激活函数
                # new_HCC = sigmoid(HCC_add)  # 激活函数
                
                new_x = mat(new_x)
                # new_HCC = mat(new_HCC)
                
                U = eye(shape(new_x.T)[0], shape(This_Y)[1])#初始化投影矩阵
                U_list=[]
                for i in range(loop_num): 
                    # print(i)
                    U=update_U_ELM_SM(U,new_x.T,This_Y,lambda_List) 
                    U_list.append(U)
                return w,first_b,U,U_list    
            def JFSSL_SM(train_X,train_Y,loop_num,lambda_List):
                (This_XT,This_Y,This_L)=build_X_and_Y_and_W_and_D_and_L(train_X,train_Y)
                U = eye(shape(This_XT)[0], shape(This_Y)[1])#初始化投影矩阵
                for i in range(loop_num):
                    U=update_U_JFSSL_SM(U,This_XT,This_Y,This_L,lambda_List) 
                return U
            def JFSSL_2M(train_X,train_Y,loop_num,feature1_num,lambda_List):
                X_List=[]
                L_List=[]
                print(shape(train_X[:,:feature1_num]))
                print(shape(train_X[:,feature1_num:]))
                (X1,Y,L1)=build_X_and_Y_and_W_and_D_and_L(train_X[:,:feature1_num],train_Y)
                (X2,Y,L2)=build_X_and_Y_and_W_and_D_and_L(train_X[:,feature1_num:],train_Y)
                
                X_List.append(X1)
                X_List.append(X2)
                #X_List.append(X3)
                L_List.append(L1)
                L_List.append(L2)
                #L_List.append(L3)
                
                LL=build_WW_and_DD_and_LL(train_X[:,:feature1_num],train_X[:,feature1_num:],train_Y)
                
                U_List=[]
                U1 = eye(feature1_num, CLASS_NUM)#初始化投影矩阵
                U2 = eye(shape(train_X)[1]-feature1_num, CLASS_NUM)#初始化投影矩阵
                #U3 = eye(FEATURE_NUM_3, CLASS_NUM)#初始化投影矩阵
                U_List.append(U1)
                U_List.append(U2)
                #U_List.append(U3)
                
                
                
                print('迭代次数：'+str(loop_num))
                for num in range(loop_num):
                    U_List=update_U_JFSSL_MM(U_List,X_List,Y,L_List,LL,lambda_List)
                
                return U_List
                
if(1==1):#主程序
    if(1==1):#数据读取
        if(1==1):#数据1
            data=sio.loadmat(PATH_R1+'X.mat') 
            x1=data['X']            
            # data=sio.loadmat(PATH_R1+'X11.mat') 
            # x11=data['X11']
            # data=sio.loadmat(PATH_R1+'X21.mat') 
            # x21=data['X21']
            # data=sio.loadmat(PATH_R1+'X22.mat') 
            # x22=data['X22']
            
            # x1=hstack((x11,x21))#
            # x1=hstack((x1,x22))#86.45
            
            data=sio.loadmat(PATH_R0+'Y.mat') 
            Y=data['Y']
            print('#######################')
            print('样本数：'+str(shape(Y)[0]))
            print('样本1特征维度：'+str(shape(x1)[1]))
            if(1==0):#数据中心化
                x1=Center_Array(x1)                
            if(1==1):#特征归一化
                if(1==1):#最大最小值归一
                    min_max_scaler = MinMaxScaler(feature_range=(-1,1))#缺省时默认（0,1）
                    x1 = min_max_scaler.fit_transform(x1)#归一化后的结果
                if(1==0):#正态归一
                    x_scaler = StandardScaler()
                    x1 = x_scaler.fit_transform(x1)#归一化后的结果
                    #(x-u)/v  每个维度的方法不同，所以归一化的尺寸不同，所以效果没有最大最小值好。    
            if(1==1):#特征降维
                if(1==1):#PCA
                    pca=PCA(n_components=PCA_NUM1)#自动'mle'
                    pca.fit(x1)
                    x1=pca.transform(x1)        
                    print(shape(x1))
                if(1==0):#KPCA
                    
                    kpca = KernelPCA(kernel='rbf',gamma=10,n_components=2)#自动'mle'
                    kpca.fit(x1)
                    x1=kpca.transform(x1)        
                    print(shape(x1))    
                if(1==0):#SVD
                    x1=SVD(x1,PCA_NUM1)
        if(1==1):#数据2
            data=sio.loadmat(PATH_R2+'X.mat') 
            x2=data['X']
            # data=sio.loadmat(PATH_R0+'Y.mat') 
            # Y=data['Y']
            
            
            # data=sio.loadmat(PATH_R3+'X.mat') 
            # x3=data['X']
            
            # data=sio.loadmat(PATH_R3+'X11.mat') 
            # x11=data['X11']
            # data=sio.loadmat(PATH_R3+'X21.mat') 
            # x21=data['X21']
            # data=sio.loadmat(PATH_R3+'X22.mat') 
            # x22=data['X22']
            # x3=hstack((x11,x21))#
            # x3=hstack((x1,x22))#86.45
            
            
            # x2=hstack((x2,x3))
            # x2=x3
            print('#######################')
            #print('样本数：'+str(shape(Y)[0]))
            print('样本2特征维度：'+str(shape(x2)[1]))
            
            if(1==0):#数据中心化
                x2=Center_Array(x2)
            
            if(1==1):#特征归一化
                if(1==1):#最大最小值归一
                    min_max_scaler = MinMaxScaler(feature_range=(-1,1))#缺省时默认（0,1）
                    x2 = min_max_scaler.fit_transform(x2)#归一化后的结果
                if(1==0):#正态归一
                    x_scaler = StandardScaler()
                    x2 = x_scaler.fit_transform(x2)#归一化后的结果
            
            
            if(1==1):#特征降维
                if(1==1):#PCA
                    pca=PCA(n_components=PCA_NUM2)#自动'mle'
                    pca.fit(x2)
                    x2=pca.transform(x2)        
                    print(shape(x2))
                if(1==0):#SVD
                    x2=SVD(x2,PCA_NUM2)
        if(1==1):#数据3
            data=sio.loadmat(PATH_R3+'X.mat') 
            x3=data['X']
            print('#######################')
            #print('样本数：'+str(shape(Y)[0]))
            print('样本3特征维度：'+str(shape(x3)[1]))
            
            if(1==0):#数据中心化
                x3=Center_Array(x3)
            
            if(1==1):#特征归一化
                if(1==1):#最大最小值归一
                    min_max_scaler = MinMaxScaler(feature_range=(-1,1))#缺省时默认（0,1）
                    x3 = min_max_scaler.fit_transform(x3)#归一化后的结果
                if(1==0):#正态归一
                    x_scaler = StandardScaler()
                    x3 = x_scaler.fit_transform(x3)#归一化后的结果
            
            
            if(1==0):#特征降维
                if(1==1):#PCA
                    pca=PCA(n_components=PCA_NUM2)#自动'mle'
                    pca.fit(x3)
                    x3=pca.transform(x3)        
                    print(shape(x3))
                if(1==0):#SVD
                    x3=SVD(x3,PCA_NUM2)            
        x=hstack((x1,x2))
        x=hstack((x,x3))
        # x=x1
        if(MSR_AS!=0):#分库
            x,Y=MSR_Divide(x,Y,MSR_AS)
        if(1==1):
            y=Y[:,0].reshape(1,ALL_SAMPLE_NUM)[0]
            print('#######################')
            print('降维分库后样本数：'+str(shape(y)[0]))
            print('降维分库后特征维度：'+str(shape(x)[1]))
    if(1==1):#数据集划分
        if(CROSS_SUBJECT==1):#交叉主题
            train_X,test_X,train_Y,test_Y=CrossSubject(x,Y)#单数subject训练,双数测试
            start = time.time()
            print(start)
            if(1==0):#存储
                
                if(1==0):#L2crc
                    if(1==0):#存储
                        PATH_W11='D:\\Desktop\\04_博士期间论文及代码\\第一篇论文\\投稿\\ChinaMM - 副本\\数据\\03_L2CRC\\data(PCA)\\'
                        F_train_size=Sample_class_size(train_X,train_Y,CLASS_NUM)
                        sio.savemat(PATH_W11+'train_X.mat', {'train_X': train_X})
                        sio.savemat(PATH_W11+'test_X.mat', {'test_X': test_X})
                        sio.savemat(PATH_W11+'F_train_size.mat', {'F_train_size': F_train_size})
                    if(1==1):#读取与分类
                        PATH_W11='D:\\Desktop\\04_博士期间论文及代码\\第一篇论文\\投稿\\ChinaMM - 副本\\数据\\03_L2CRC\\data(PCA)\\'
                        data=sio.loadmat(PATH_W11+'ty.mat') 
                        ty=data['ty'][0]
                        # print(ty)
                        mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                            
                        (a,b)=Confusion_Matrix_a(mat)
                        
                        print(b)#总体识别率   
                if(1==1):#ccca
                    PATH_W11='D:\\Desktop\\04_博士期间论文及代码\\第一篇论文\\投稿\\ChinaMM - 副本\\数据\\01_CCCA\\data(PCA)\\'
                    train_X1=train_X[:,:PCA_NUM1]
                    train_X2=train_X[:,PCA_NUM1:]
                    print(shape(train_X1))
                    train_XX1,AA=Sample_Mean(train_X1,train_Y,CLASS_NUM)
                    print(shape(train_XX1))
                    train_XX2,AA=Sample_Mean(train_X2,train_Y,CLASS_NUM)
                    # sio.savemat(PATH_W11+'train_XX1.mat', {'train_XX1': train_XX1})
                    # sio.savemat(PATH_W11+'train_XX2.mat', {'train_XX2': train_XX2})
                    # sio.savemat(PATH_W11+'train_X1.mat', {'train_X1': train_X1})
                    # sio.savemat(PATH_W11+'train_X2.mat', {'train_X2': train_X2})
                    
                    # test_X1=test_X[:,:PCA_NUM1]
                    # test_X2=test_X[:,PCA_NUM1:]
                    
                    # sio.savemat(PATH_W11+'test_X1.mat', {'test_X1': test_X1})
                    # sio.savemat(PATH_W11+'test_X2.mat', {'test_X2': test_X2})
                    # sio.savemat(PATH_W11+'train_Y.mat', {'train_Y': train_Y})
                    # sio.savemat(PATH_W11+'test_Y.mat', {'test_Y': test_Y})
                    print (time.time()-start)
                    print(time.time())
                if(1==0):#cca
                    PATH_W11='D:\\Desktop\\04_博士期间论文及代码\\第一篇论文\\投稿\\ChinaMM - 副本\\数据\\01_CCCA\\data(PCA)\\'
                    train_X1=train_X[:,:PCA_NUM1]
                    train_X2=train_X[:,PCA_NUM1:]
                    sio.savemat(PATH_W11+'train_X1.mat', {'train_X1': train_X1})
                    sio.savemat(PATH_W11+'train_X2.mat', {'train_X2': train_X2})
                
                    test_X1=test_X[:,:PCA_NUM1]
                    test_X2=test_X[:,PCA_NUM1:]
                    
                    sio.savemat(PATH_W11+'test_X1.mat', {'test_X1': test_X1})
                    sio.savemat(PATH_W11+'test_X2.mat', {'test_X2': test_X2})
                    sio.savemat(PATH_W11+'train_Y.mat', {'train_Y': train_Y})
                    sio.savemat(PATH_W11+'test_Y.mat', {'test_Y': test_Y})
            if(1==0):#读取
                
                PATH_W11='D:\\Desktop\\04_博士期间论文及代码\\第一篇论文\\投稿\\ChinaMM - 副本\\数据\\01_CCCA\\data(PCA)\\'
                data=sio.loadmat(PATH_W11+'train_X.mat') 
                train_X=data['train_X'] 
                print(shape(train_X))
                data=sio.loadmat(PATH_W11+'test_X.mat') 
                test_X=data['test_X']
                print(shape(test_X))
            
                
        else:#随机
            train_X,test_X,train_Y,test_Y=train_test_split(x, y,test_size=TEST_NUM/ALL_SAMPLE_NUM,random_state=2)       
        print('#######################')
        print('训练样本数：'+str(shape(train_X)[0]))
        print('特征维度：'+str(shape(train_X)[1]))
        print('#######################')
        
        
    if(1==1):#特征投影融合        
        if(1==1):#子空间投影类
            if(1==0):#合SL
                print('#######################')
                print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                ###子空间投影矩阵训练
                #U1=SM_SC_SubspaceLearning(test_X,test_Y,LOOP_NUM)
                #train_X_1=dot(train_X,U1)
                #test_X_1=dot(test_X,U1)
                #(train_X_mean,train_mean_Y)=Sample_Mean(train_X,train_Y,CLASS_NUM)
                lambda_List=[]
                lambda_List.append(200)#参数1
                U=SL_SM(train_X,train_Y,LOOP_NUM,lambda_List)
                train_X=dot(train_X,U)
                test_X=dot(test_X,U)

                print('子空间投影后特征维度：'+str(shape(train_X)[1]))
                print('#######################')
                if(1==0):#分类
                    ty=[]
                    for i in range(len(test_X)):
                        ty.append(argmax(test_X[i])+1)
                    #print(ty)
                    mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                    print(mat)
                    (a,b)=Confusion_Matrix_a(mat)
                    print(array(a))#每种行为识别率
                    print(b)#总体识别率 
            if(1==0):#分SL
                print('#######################')
                print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                
                lambda_List=[]
                lambda_List.append(500)#参数1
                lambda_List.append(500)#参数2
                U1=SL_SM(train_X[:,:PCA_NUM1],train_Y,LOOP_NUM,lambda_List)
                train_X1=dot(train_X[:,:PCA_NUM1],U1)
                test_X1=dot(test_X[:,:PCA_NUM1],U1)
                
                U2=SL_SM(train_X[:,PCA_NUM1:],train_Y,LOOP_NUM,lambda_List)
                train_X2=dot(train_X[:,PCA_NUM1:],U2)
                test_X2=dot(test_X[:,PCA_NUM1:],U2)
                
                train_X=hstack((train_X1,train_X2))
                test_X=hstack((test_X1,test_X2))
                print('子空间投影后特征维度：'+str(shape(train_X)[1]))
                print('#######################')
            if(1==0):#JFSSL
                lambda_List=[]
                lambda_List.append(200)#参数1
                lambda_List.append(0)#参数2：0.069(0.062-0.07)
                #lambda2=0:同分SL
                U_List=JFSSL_2M(train_X,train_Y,LOOP_NUM,PCA_NUM1,lambda_List)
                train_X1=dot(train_X[:,:PCA_NUM1],U_List[0])
                train_X2=dot(train_X[:,PCA_NUM1:],U_List[1])

                
                test_X1=zeros((shape(test_X)[0],PCA_NUM1))
                test_X2=zeros((shape(test_X)[0],PCA_NUM2))
                
                test_X1=dot(test_X[:,:PCA_NUM1],U_List[0])
                test_X2=dot(test_X[:,PCA_NUM1:],U_List[1])

                
                train_X=hstack((train_X1,train_X2))
                test_X=hstack((test_X1,test_X2))
                print('子空间投影后特征维度：'+str(shape(train_X)[1]))
                print('#######################')
                # print(test_X)
                if(1==0):#分类
                    print(shape(test_X1))
                    test_X=mat(test_X1)+mat(test_X2)
                    print(shape(test_X))
                    ty=[]
                    for i in range(len(test_X)):
                        ty.append(argmax(test_X[i])+1)
                    
                    mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                    print(mat)
                    (a,b)=Confusion_Matrix_a(mat)
                    print(array(a))#每种行为识别率
                    print(b)#总体识别率 
            if(1==0):#全局CBBMC
                if(1==1):#融合
                    print('#######################')
                    print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                    ###子空间投影矩阵训练
                    lambda_List0=[]
                    lambda_List0.append(1000)#参数1稀疏项：200
                    lambda_List0.append(0.1)#参数2：10e12(10e11-10e13)
                    loop_num0=5#迭代次数:30
                    #(200,10e12,30):93.77
                    #lambda2=0:同合SL
                    U,U_list=DDSL_SM(train_X,train_Y,loop_num0,lambda_List0)
                    # train_X=dot(train_X,U)
                    
                    print('子空间投影后特征维度：'+str(shape(train_X)[1]))
                    print('#######################')
                    if(1==0):
                        for m in range(loop_num0):
                            # print(shape(dot(test_X[:,:PCA_NUM1],U1_list[m])))
                            test_XS=dot(test_X,U_list[m])
                            # print(shape(test_X1))
                            ty=[]
                            for i in range(len(test_XS)):
                                ty.append(argmax(test_XS[i])+1)
                            
                            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                            
                            (a,b)=Confusion_Matrix_a(mat)
                            
                            print(str(m+1)+':'+str(b))#总体识别率
                if(1==1):#分类
                        test_XS=dot(test_X,U)
                        ty=[]
                        for i in range(len(test_XS)):
                            ty.append(argmax(test_XS[i])+1)
                        #print(ty)
                        mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                        print(mat)
                        (a,b)=Confusion_Matrix_a(mat)
                        print(array(a))#每种行为识别率
                        print(b)#总体识别率 
                        # print(metrics.calinski_harabasz_score(test_XS,test_Y))
            if(1==1):#局部CBBMC
                if(1==1):#融合
                    print('#######################')
                    print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                    ###子空间投影矩阵训练
                    if(1==1):#特征1
                        lambda_List1=[]
                        lambda_List1.append(6.3)#参数1:50-80
                        lambda_List1.append(0)#参数2
                        loop_num1=5#迭代次数:30
                        U1,U1_list=DDSL_SM(train_X[:,:PCA_NUM1],train_Y,loop_num1,lambda_List1)
                        if(1==0):
                            for m in range(loop_num1):
                                # print(shape(dot(test_X[:,:PCA_NUM1],U1_list[m])))
                                test_X1=dot(test_X[:,:PCA_NUM1],U1_list[m])
                                # print(shape(test_X1))
                                ty=[]
                                for i in range(len(test_X1)):
                                    ty.append(argmax(test_X1[i])+1)
                                
                                mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                                
                                (a,b)=Confusion_Matrix_a(mat)
                                
                                print(str(m+1)+':'+str(b))#总体识别率
                    if(1==1):#特征2
                        lambda_List2=[]
                        lambda_List2.append(200)#参数1:300
                        lambda_List2.append(0.1)#参数2:0.1
                        loop_num2=1#迭代次数:30
                        U2,U2_list=DDSL_SM(train_X[:,PCA_NUM1:],train_Y,loop_num2,lambda_List2)
                        if(1==0):
                            for m in range(loop_num2):
                                # print(shape(dot(test_X[:,:PCA_NUM1],U1_list[m])))
                                test_X2=dot(test_X[:,PCA_NUM1:],U2_list[m])
                                # print(shape(test_X1))
                                ty=[]
                                for i in range(len(test_X2)):
                                    ty.append(argmax(test_X2[i])+1)
                                
                                mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                                
                                (a,b)=Confusion_Matrix_a(mat)
                                
                                print(str(m+1)+':'+str(b))#总体识别率
                if(1==1):#分类
                    # print(shape(test_X))
                    # print(PCA_NUM1)
                    test_X1=dot(test_X[:,:PCA_NUM1],U1_list[loop_num1-1])
                    test_X2=dot(test_X[:,PCA_NUM1:],U2_list[loop_num2-1])
                    if(1==1):
                        test_X=mat(test_X1)+1.1*mat(test_X2)
                        # test_X=np.multiply(mat(test_X1),mat(test_X2))
                        # test_X=np.maximum(mat(test_X1),mat(test_X2))
                    # test_X=mat(test_X1)+mat(test_X2)+mat(test_X0)
                    # test_X=mat(test_X1)
                    print(shape(test_X))
                    ty=[]
                    for i in range(len(test_X)):
                        ty.append(argmax(test_X[i])+1)
                    
                    mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                    print(mat)
                    (a,b)=Confusion_Matrix_a(mat)
                    print(array(a))#每种行为识别率
                    print(b)#总体识别率 
            if(1==0):#全局局部DDSL
                if(1==1):#融合
                    if(1==1):#特征0
                        print('#######################')
                        print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                        ###子空间投影矩阵训练
                        lambda_List0=[]
                        lambda_List0.append(200)#参数1稀疏项：200
                        lambda_List0.append(0.1)#参数2：10e12(10e11-10e13)
                        loop_num0=1#迭代次数:30
                        #(200,10e12,30):93.77
                        #lambda2=0:同合SL
                        U0,U0_list=DDSL_SM(train_X,train_Y,loop_num0,lambda_List0)
                        # train_X=dot(train_X,U)
                    if(1==1):#特征1
                        lambda_List1=[]
                        lambda_List1.append(60)#参数1:50-80
                        lambda_List1.append(0.1)#参数2
                        loop_num1=1#迭代次数:30
                        U1,U1_list=DDSL_SM(train_X[:,:PCA_NUM1],train_Y,loop_num1,lambda_List1)
                    if(1==1):#特征2
                        lambda_List2=[]
                        lambda_List2.append(300)#参数1:300
                        lambda_List2.append(0.1)#参数2:0.1
                        loop_num2=1#迭代次数:30
                        U2,U2_list=DDSL_SM(train_X[:,PCA_NUM1:],train_Y,loop_num2,lambda_List2)
                    if(1==1):#融合方式  
                        if(1==0):#级联然后子空间选择
                            train_X0=dot(train_X,U0)    
                            train_X1=dot(train_X[:,:PCA_NUM1],U1)
                            train_X2=dot(train_X[:,PCA_NUM1:],U2)
                            train_XX=hstack((train_X0,train_X1))#
                            train_XX=hstack((train_XX,train_X2))#
                            ###子空间投影矩阵训练
                            lambda_List0=[]
                            lambda_List0.append(10)#参数1稀疏项：200
                            lambda_List0.append(0.01)#参数2：10e12(10e11-10e13)
                            loop_num0=2#迭代次数:30
                            #(200,10e12,30):93.77
                            #lambda2=0:同合SL
                            UU,UU_list=DDSL_SM(train_XX,train_Y,loop_num0,lambda_List0)
                            test_X0=dot(test_X,U0)    
                            test_X1=dot(test_X[:,:PCA_NUM1],U1)
                            test_X2=dot(test_X[:,PCA_NUM1:],U2)
                            test_XX=hstack((test_X0,test_X1))#
                            test_XX=hstack((test_XX,test_X2))#
                            test_XS=dot(test_XX,UU)
                        if(1==0):#相加
                            test_X0=dot(test_X,U0)    
                            test_X1=dot(test_X[:,:PCA_NUM1],U1)
                            test_X2=dot(test_X[:,PCA_NUM1:],U2)
                            test_XS=mat(test_X0)+mat(test_X1)+mat(test_X2)
                        if(1==0):#相乘
                            test_X0=dot(test_X,U0)    
                            test_X1=dot(test_X[:,:PCA_NUM1],U1)
                            test_X2=dot(test_X[:,PCA_NUM1:],U2)
                            test_XS=multiply(mat(test_X0),mat(test_X1))
                            test_XS=multiply(mat(test_XS),mat(test_X2))  
                        if(1==0):#最大值
                            test_X0=dot(test_X,U0)    
                            test_X1=dot(test_X[:,:PCA_NUM1],U1)
                            test_X2=dot(test_X[:,PCA_NUM1:],U2)
                            test_XS=maximum(mat(test_X0),mat(test_X1))
                            test_XS=maximum(mat(test_XS),mat(test_X2))    
                    test_XS=test_X0=dot(test_X,U0)         
                    ty=[]
                    for i in range(len(test_XS)):
                        ty.append(argmax(test_XS[i])+1)
                    #print(ty)
                    mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                    print(mat)
                    (a,b)=Confusion_Matrix_a(mat)
                    print(array(a))#每种行为识别率
                    print(b)#总体识别率
       
            if(1==0):#级FFSL
                if(1==1):#融合
                    print('#######################')
                    print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                    ###子空间投影矩阵训练
                    lambda_List0=[]
                    lambda_List0.append(200)#参数1稀疏项：200
                    lambda_List0.append(0.1)#参数2：10e12(10e11-10e13)
                    loop_num0=3#迭代次数:30
                    num_hidden=20000
                    #(200,10e12,30):93.77
                    #lambda2=0:同合SL
                    W,B,U,U_list=FFSL_SM(train_X,train_Y,num_hidden,loop_num0,lambda_List0)
                    # train_X=dot(train_X,U)
                    First_B=B
                    for i in range(shape(test_X)[0] - 1):
                        B = np.row_stack((B, First_B))  # row_stack 以叠加行的方式填充数组
                    print('子空间投影后特征维度：'+str(shape(train_X)[1]))
                    print('#######################')
                    if(1==1):
                        for m in range(loop_num0):
                            # print(shape(dot(test_X[:,:PCA_NUM1],U1_list[m])))
                            
                            
                            test_X_mul = np.dot(test_X, W)  # 输入乘以权重
    
                            
                            test_X_add = test_X_mul + B  # 加偏置

                            new_test_X = sigmoid(test_X_add)  # 激活函数
                            
                            
                            test_XS=dot(new_test_X,U_list[m])
                            # print(shape(test_X1))
                            ty=[]
                            for i in range(len(test_XS)):
                                ty.append(argmax(test_XS[i])+1)
                            
                            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                            
                            (a,b)=Confusion_Matrix_a(mat)
                            
                            print(str(m+1)+':'+str(b))#总体识别率
                if(1==0):#分类
                        test_XS=dot(test_X,U)
                        ty=[]
                        for i in range(len(test_XS)):
                            ty.append(argmax(test_XS[i])+1)
                        #print(ty)
                        mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                        print(mat)
                        (a,b)=Confusion_Matrix_a(mat)
                        print(array(a))#每种行为识别率
                        print(b)#总体识别率
            if(1==0):#级HHSL同极限学习机
                if(1==1):#融合
                    print('#######################')
                    print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                    ###子空间投影矩阵训练
                    lambda_List0=[]
                    lambda_List0.append(200)#参数1稀疏项：200
                    lambda_List0.append(0.1)#参数2：10e12(10e11-10e13)
                    loop_num0=1#迭代次数:30
                    num_hidden=5000
                    #(200,10e12,30):93.77
                    #lambda2=0:同合SL
                    W,B,U,U_list=HHSL_SM(train_X,train_Y,num_hidden,loop_num0,lambda_List0)
                    # train_X=dot(train_X,U)
                    First_B=B
                    for i in range(shape(test_X)[0] - 1):
                        B = np.row_stack((B, First_B))  # row_stack 以叠加行的方式填充数组
                    print('子空间投影后特征维度：'+str(shape(train_X)[1]))
                    print('#######################')
                    if(1==1):
                        for m in range(loop_num0):
                            # print(shape(dot(test_X[:,:PCA_NUM1],U1_list[m])))
                            
                            
                            test_X_mul = np.dot(test_X, W)  # 输入乘以权重
    
                            
                            test_X_add = test_X_mul + B  # 加偏置

                            new_test_X = sigmoid(test_X_add)  # 激活函数
                            
                            
                            test_XS=dot(new_test_X,U_list[m])
                            # print(shape(test_X1))
                            ty=[]
                            for i in range(len(test_XS)):
                                ty.append(argmax(test_XS[i])+1)
                            
                            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                            
                            (a,b)=Confusion_Matrix_a(mat)
                            
                            print(str(m+1)+':'+str(b))#总体识别率
                            # print(metrics.calinski_harabasz_score(test_XS,test_Y)) 
                if(1==0):#分类
                        test_XS=dot(test_X,U)
                        ty=[]
                        for i in range(len(test_XS)):
                            ty.append(argmax(test_XS[i])+1)
                        #print(ty)
                        mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                        print(mat)
                        (a,b)=Confusion_Matrix_a(mat)
                        print(array(a))#每种行为识别率
                        print(b)#总体识别率
                        # print(metrics.calinski_harabasz_score(test_XS,test_Y))                        
            if(1==0):#分DDSL
                if(1==1):#融合
                    print('#######################')
                    print('子空间投影前特征维度：'+str(shape(train_X)[1]))
                    ###子空间投影矩阵训练
                    if(1==1):#特征1
                        lambda_List1=[]
                        lambda_List1.append(65)#参数1:50-80
                        lambda_List1.append(0.05)#参数2
                        loop_num1=10#迭代次数:30
                        U1,U1_list=DDSL_SM(train_X[:,:PCA_NUM1],train_Y,loop_num1,lambda_List1)
                        if(1==0):
                            for m in range(loop_num1):
                                # print(shape(dot(test_X[:,:PCA_NUM1],U1_list[m])))
                                test_X1=dot(test_X[:,:PCA_NUM1],U1_list[m])
                                # print(shape(test_X1))
                                ty=[]
                                for i in range(len(test_X1)):
                                    ty.append(argmax(test_X1[i])+1)
                                
                                mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                                
                                (a,b)=Confusion_Matrix_a(mat)
                                
                                print(str(m+1)+':'+str(b))#总体识别率
                    if(1==1):#特征2
                        lambda_List2=[]
                        lambda_List2.append(300)#参数1:300
                        lambda_List2.append(0.1)#参数2:0.1
                        loop_num2=5#迭代次数:30
                        U2,U2_list=DDSL_SM(train_X[:,PCA_NUM1:],train_Y,loop_num2,lambda_List2)
                        if(1==0):
                            for m in range(loop_num2):
                                # print(shape(dot(test_X[:,:PCA_NUM1],U1_list[m])))
                                test_X2=dot(test_X[:,PCA_NUM1:],U2_list[m])
                                # print(shape(test_X1))
                                ty=[]
                                for i in range(len(test_X2)):
                                    ty.append(argmax(test_X2[i])+1)
                                
                                mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                                
                                (a,b)=Confusion_Matrix_a(mat)
                                
                                print(str(m+1)+':'+str(b))#总体识别率
                if(1==1):#分类
                    # print(shape(test_X))
                    test_X1=dot(test_X[:,:PCA_NUM1],U1_list[2])
                    test_X2=dot(test_X[:,PCA_NUM1:],U2_list[2])
                    test_X=mat(test_X1)+mat(test_X2)
                    # test_X=mat(test_X1)+mat(test_X2)+mat(test_X0)
                    # test_X=mat(test_X1)
                    print(shape(test_X))
                    ty=[]
                    for i in range(len(test_X)):
                        ty.append(argmax(test_X[i])+1)
                    
                    mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
                    print(mat)
                    (a,b)=Confusion_Matrix_a(mat)
                    print(array(a))#每种行为识别率
                    print(b)#总体识别率
    if(1==0):#融合评价
        if(1==0):#定量
            if(1==1):#特征归一化
                if(1==1):#最大最小值归一
                    min_max_scaler = MinMaxScaler(feature_range=(0,1))#缺省时默认（0,1）
                    test_XS = min_max_scaler.fit_transform(test_XS)#归一化后的结果
                if(1==0):#正态归一
                    x_scaler = StandardScaler()
                    test_XS = x_scaler.fit_transform(test_XS)#归一化后的结果
            # print(test_XS)
            print(metrics.silhouette_score(test_XS,test_Y, metric='euclidean'))#轮廓系数越大越好
            print(metrics.calinski_harabasz_score(test_XS,test_Y))  #CH指数越大越好
            print(metrics.davies_bouldin_score(test_XS,test_Y)) #DBI值越小越好
            # print(test_Y)
        if(1==0):#定性可视化
            color_set=['b','g','r','c','m','y','k','#DDA0DD','#6495ED','#DC143C','#B8860B','#A9A9A9','#BDB76B','#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#556B2F']
            c_list=[]#与便签test_Y对应的颜色
            print(test_Y)
            print(len(test_Y))
            for i in range(len(test_Y)):
                
                c_list.append(color_set[test_Y[i]-1])
            
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(test_XS)
            X_pca = PCA(n_components=2).fit_transform(test_XS)
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_list)
            plt.legend()
            plt.subplot(122)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=c_list)
            plt.legend()
            plt.show()
    if(1==0):#分类
        if(1==1):#SVM
        
            # train_X,test_X,train_y,test_y=train_test_split(x, y,test_size=0.1,random_state=0)
            clf=svm.SVC(kernel='sigmoid', degree=5,C=10)  ##默认参数：kernel='sigmoid'\kernel='rbf'\kernel='linear'\kernel='poly', degree=3#核函数
            clf.fit(train_X,train_Y)
            
            ty = clf.predict(test_X)
            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
            print(mat)
            (a,b)=Confusion_Matrix_a(mat)
            print(array(a))#每种行为识别率
            print(b)#总体识别率
        if(1==0):#极限学习机
            #使用0123标签
            ELM = SingeHiddenLayer(train_X, train_Y-1, 10000)  # 训练数据集，训练集的label，隐藏层节点个数
            ELM.train(train_X, train_Y-1, CLASS_NUM)
            ELM.predict(test_X)
            ELM.score(test_Y-1)    
        if(1==0):#KNN
            # train_X,test_X,train_y,test_y=train_test_split(x, y,test_size=0.5,random_state=0)
            knn = neighbors.KNeighborsClassifier() #取得knn分类器    

            knn.fit(train_X,train_Y)
               

            ty = knn.predict(test_X)
            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
            print(mat)
            (a,b)=Confusion_Matrix_a(mat)
            print(array(a))#每种行为识别率
            
            print(b)#总体识别率
        if(1==0):#随机森林
            clf = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=4)#200棵数
            rf_clf = clf.fit(train_X,train_Y)
            ty = rf_clf.predict(test_X)
            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
            print(mat)
            (a,b)=Confusion_Matrix_a(mat)
            print(array(a))#每种行为识别率
            print(b)#总体识别率
        if(1==0):#高斯贝叶斯 
            gnb = Pipeline([
            ('sc', StandardScaler()),
            ('clf', GaussianNB())])
            gnb.fit(train_X,train_Y)
            ty = gnb.predict(test_X)
            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
            print(mat)
            (a,b)=Confusion_Matrix_a(mat)
            print(array(a))#每种行为识别率
            print(b)#总体识别率
        if(1==0):#强化学习
            data_train = xgb.DMatrix(train_X, label=train_Y)
            data_test = xgb.DMatrix(test_X, label=test_Y)

            watch_list = [(data_test, 'eval'), (data_train, 'train')]
            # param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 18}
            param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 19}
            bst = xgb.train(param, data_train, num_boost_round=2, evals=watch_list)
            P = bst.get_fscore()
            ty = bst.predict(data_test)
            mat=confusion_matrix(test_Y,ty)#生成每折混淆矩阵
            print(mat)
            (a,b)=Confusion_Matrix_a(mat)
            print(array(a))#每种行为识别率
            print(b)#总体识别率
        if(1==0):#神经网络    
            y_train_label = LabelBinarizer().fit_transform(train_Y)
            print(y_train_label)
            y_test_label = LabelBinarizer().fit_transform(test_Y)##输出010编码
            train_num,feature_num=shape(train_X)
            classify = NeuralNetwork(feature_num,100,CLASS_NUM)
            loss_list,accuracy_train,accuracy_val = classify.train(train_X,y_train_label
                                                       ,train_Y,test_X,test_Y)
            
            ty = classify.predict(test_X)##输出0123编码
            # print(ty)
            mat=confusion_matrix(test_Y,ty+1)#生成每折混淆矩阵
            print(mat)
            (a,b)=Confusion_Matrix_a(mat)
            print(array(a))#每种行为识别率
            print(b)#总体识别率
            # accuracy = np.mean(ty == test_y-1)
            # print("the accuracy is "+str(accuracy))
    
    end =  time.time()  
    # print (end-start).days # 0 天数
    # print ('耗时：')
    # print(end-start)# 30.029522 精确秒数
    # print (end-start).seconds # 30 秒数
    # print (end-start).microseconds # 29522 毫秒数