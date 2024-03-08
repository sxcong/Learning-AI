import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


def load_data(path, files):
    import gzip  ###解压缩gz文件
    paths = [path + each for each in files]
    with gzip.open(paths[0],'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)  #frombuffer将data以流的形式读入转化成ndarray对象 #第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
    with gzip.open(paths[1],'rb') as imgpath:
        train_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)
    with gzip.open(paths[2],'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3],'rb') as imgpath:
        test_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_labels), 28, 28)
    return (train_images, train_labels), (test_images, test_labels)
    


if __name__ == '__main__':
	print('Tensorflow Version: {}'.format(tf.__version__))


	#LINUX可以下载,windows下载失败，所以先下载再使用
	#(train_image, train_lable), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

	path = "./FashionMNIST/raw/"
	files = ['train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz']
	(train_images, train_labels), (test_images, test_labels) = load_data(path, files)


	print(train_images.shape)
	print(train_labels.shape)
	#plt.imshow(train_images[3])
	#plt.show()

	#每张图片像素的取值范围是0-255，需要做归一化处理。
	train_images = train_images/255
	test_images = test_images/255
	#构建模型
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # 28*28
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))

	#查看模型
	print(model.summary())

	#编译模型
	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
	#训练模型
	model.fit(train_images, train_labels, epochs=3)

	#在测试数据集上该评估模型
	ret = model.evaluate(test_images, test_labels, verbose=0)
	print(ret)

	#保存整个模型
	model.save('less_model.h5')


	#保存模型的目的在于后期方便对模型进行载入，下面载入模型，并查看模型。

	new_model = tf.keras.models.load_model('less_model.h5')
	new_model.summary()

	#编译模型
	new_model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['acc'])

	#在测试集上评估该模型
	ret = new_model.evaluate(test_images, test_labels, verbose=0)
	print(ret)