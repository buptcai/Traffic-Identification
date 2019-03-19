from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import cv2
import numpy as np

category_lists = ['biaoPai','gaiLiang','gongShangCeQiang','huLan','huPo',
				  'liang','qiaoDun','qiaoMian','qiaoTai','shenSuoFeng',
				  'xieshuiKong','yiQiang','zhiZuo','zhuGongQuan','zhuiPo']

def build_model():
	base_model = InceptionV3(weights = 'imagenet',include_top = False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024,activation = 'relu')(x)
	predictions = Dense(15,activation = 'softmax')(x)
	model = Model(inputs = base_model.input,outputs = predictions)
	#model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ["accuracy"])
	return model

def load_model(weights_path):
	weights_path = weights_path
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.9
	session = tf.Session(config = config)
	KTF.set_session(session)
	model = build_model()
	model.load_weights(weights_path)
	return model

def load_image(filepath):
	image = cv2.imread(filepath)
	image = cv2.resize(image, (224,224))
	data = np.array(image,dtype = 'float')/255.0
	data = data.reshape((1,224,224,3))
	return data

def predict(model,data):
	result = model.predict(data)
	acc = np.max(result)
	pos = np.argmax(result)
	category = category_lists[pos]
	return acc,category