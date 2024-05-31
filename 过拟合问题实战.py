import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers
from keras.src import regularizers
from keras.src.layers import Dense
from matplotlib import cm
from sklearn.datasets import make_moons
import keras
from sklearn.model_selection import train_test_split

NSamples=1000
TEST_SIZE=200
X,y=make_moons(n_samples=NSamples,noise=0.25,random_state=100)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,random_state=42)
#构造训练数据
XX = np.linspace(-3, 3, 1000)
YY = np.linspace(-2, 2, 1000)
XX,YY=np.meshgrid(XX, YY)

def make_plot(X,y,XX=None,YY=None,preds=None):
    plt.figure()
    axes=plt.gca() #获得图形的轴对象
    axes.set_xlim([-3,3])
    axes.set_ylim([-2,2])
    #绘制预测曲面
    if(XX is not None and preds is not None):
        plt.contourf(XX,YY,preds.reshape(XX.shape),25,alpha=0.8,cmap=cm.Spectral)
    #绘制正负样本
    else:
        markers = ['o' if i == 1 else 's' for i in y.ravel()]
        colors = ['red' if i == 1 else 'blue' for i in y.ravel()]
        for i in range(len(X)):
            axes.scatter(X[i, 0], X[i, 1], c=colors[i], s=20, edgecolors='none'
                         , marker=markers[i])  # 点大小为20，设置颜色映射，不显示边缘的颜色，标记点的形状
    plt.pause(2)
#make_plot(X,y)


#网络层数的影响
"""
for n in range(5):
    #模型初始化
    model=keras.Sequential()
    model.add(Dense(8,activation='relu'))
    for _ in range(n):
        model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #模型训练
    history = model.fit(X_train, y_train, epochs=30,verbose=1)
    #计算预测值
    preds=model.predict(np.c_[XX.ravel(),YY.ravel()]) #preds是1000,000*1的张量
    print(preds.shape)

    #绘图
    plt.figure()
    axes = plt.gca()  # 获得图形的轴对象
    axes.set_xlim([-3, 3])
    axes.set_ylim([-2, 2])
    # 绘制预测曲面
    plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.8, cmap=cm.Spectral)
    plt.pause(1)
"""

#Dropout的影响
"""
for n in range(5):
    model=keras.Sequential()
    model.add(Dense(8,activation='relu'))
    counter=0
    #模型初始化
    for _ in range(5):
        model.add(Dense(64,activation='relu'))
        if counter<n:
            counter+=1
            model.add(layers.Dropout(rate=0.5)) #当前网络层丢弃50%的数据
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 训练
    history = model.fit(X_train, y_train, epochs=30, verbose=1)
    # 计算预测值
    preds = model.predict(np.c_[XX.ravel(), YY.ravel()])  # preds是1000,000*1的张量
    print(preds.shape)

    # 绘图
    plt.figure()
    axes = plt.gca()  # 获得图形的轴对象
    axes.set_xlim([-3, 3])
    axes.set_ylim([-2, 2])
    # 绘制预测曲面
    plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.8, cmap=cm.Spectral)
    plt.pause(1)
"""


#正则化的影响
def build_model_with_regularization(_lambda):
    model=keras.Sequential()
    model.add(Dense(8,input_dim=2,activation='relu'))
    model.add(Dense(256,activation='relu',
                    kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dense(256,activation='relu',
                    kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dense(256,activation='relu',
                    kernel_regularizer=regularizers.l2(_lambda)))
    #输出层
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

for _lambda in [1e-5,1e-3,1e-1,0.12,0.13]:
    model=build_model_with_regularization(_lambda)
    #print(X_train.shape,y_train.shape)
    history=model.fit(X_train,y_train,epochs=30,verbose=1)
    preds=model.predict(np.c_[XX.ravel(),YY.ravel()])
    #绘图
    make_plot(X,y,XX,YY,preds)