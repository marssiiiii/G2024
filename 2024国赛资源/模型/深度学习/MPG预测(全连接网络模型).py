import tensorflow as tf
import keras
import pandas as pd
from keras import layers
from keras.src import losses
import numpy as np
import matplotlib.pyplot as plt

#数据导入与提取
file_path="auto-pg.data"
column_name = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight",
               "Acceleration", "Model_Year", "Origin"]
dataset=pd.read_csv(file_path,names=column_name,delim_whitespace=True,
                 na_values="?",comment="\t",skipinitialspace=True,index_col=False)

origin=dataset.pop("Origin") #Origin本质为分类数据，不可直接用数值代替
dataset['USA']=(origin==1)*1.0
dataset['Europe']=(origin==2)*1.0
dataset['Japan']=(origin==3)*1.0

#切分训练集与测试集
train_dataset=dataset.sample(frac=0.8,random_state=0)
test_dataset=dataset.drop(train_dataset.index)

#将分类的标签提取出来
train_labels=train_dataset.pop('MPG')
test_labels=test_dataset.pop('MPG')

#将输入特征数据标准化
train_stats=train_dataset.describe()
train_stats=train_stats.transpose()
def norm(x):
    return(x-train_stats['mean'])/train_stats['std']
normed_train_data=norm(train_dataset)
normed_test_data=norm(test_dataset)
def fill_na_with_row_mean(row): #填充nan值
    return row.fillna(row.mean())
normed_train_data=normed_train_data.apply(fill_na_with_row_mean,axis=1)
normed_test_data=normed_test_data.apply(fill_na_with_row_mean,axis=1)

#构造训练集
train_db=(tf.data.Dataset.from_tensor_slices
          ((normed_train_data.values,train_labels.values)))
train_db=train_db.shuffle(100).batch(32)


#训练
#构造训练模型
class Network(keras.Model):
    def __init__(self):
        super(Network,self).__init__()
        self.fc1=layers.Dense(64,activation='relu')
        self.fc2=layers.Dense(64,activation='relu')
        self.fc3=layers.Dense(1)

    def call(self,inputs,training=None,mask=None):
        x=self.fc1(inputs)
        x=self.fc2(x)
        x=self.fc3(x)
        return x
#训练
model=Network()
model.build(input_shape=(None,9))
optimizer=keras.optimizers.RMSprop(0.00005) #创建优化器
times=500
loss_values=np.empty(times)
epochs=range(0,times)
for epoch in range(times):
    for step,(x,y) in enumerate(train_db):
        y=tf.expand_dims(y,1)
        with tf.GradientTape() as tape:
            out=model(x)
            loss=tf.reduce_mean(losses.MSE(y,out))
            #print(f"loss:{loss}")
            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
    print(f"loss:{loss}")
    loss_values[epoch]=loss

plt.plot(epochs,loss_values)
plt.show()


