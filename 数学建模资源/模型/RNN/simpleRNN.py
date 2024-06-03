import tensorflow as tf
from keras import layers

x=tf.random.normal([4,80,100])
cell=layers.SimpleRNNCell(64)
h=[tf.zeros([4,64])]

for xt in tf.unstack(x,axis=1):
    out,h=cell(xt,h)

#多层连接
cell0=layers.SimpleRNNCell(64)
cell1=layers.SimpleRNNCell(64)
middle_sequence=[]
h0=tf.zeros([4,64])

for xt in tf.unstack(x,axis=1):
    out0,h0=cell0(xt,h0)
    middle_sequence.append(out0)

h1=tf.zeros([4,64])
for xt in middle_sequence:
    out1,h1=cell1(xt,h1)

