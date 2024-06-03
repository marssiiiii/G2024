import tensorflow as tf
#直接对张量数值进行限制
a=tf.random.uniform([2,2])
tf.clip_by_value(a,0,4,0.6)

#限制局部范数(w/norm)*max
a=tf.random.uniform([2,2])*5
b=tf.clip_by_norm(a,5) #限制最大值为5

#限制全局范数(w*max)/max(global_norm,max) (若global_norm大于max,会进行全局缩放)
w1=tf.random.normal([3,3])
w2=tf.random.normal([3,3])

(ww1,ww2),global_norm=tf.clip_by_global_norm([w1,w2],2) #

