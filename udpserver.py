from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import cv2
import numpy as np
import socket

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

def load_model():
	weights_path = "./best_result/model-ep029-loss0.035-val_acc0.814-val_loss0.808.h5"
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

def predict():
	model = load_model()
	while True:
		filepath = input("path:")
		print('finish')
		image = load_image(filepath)
		result = model.predict(image)
		print(result)

def udpserver():
	model = load_model()
	print('Model has been intialized successfully')
	s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	s.bind(('127.0.0.1',9999))
	print('bind udp server on 127.0.0.1:9999')
	while True:
		try:
			data,addr = s.recvfrom(1024)
			filepath = data.decode()
			if(os.path.exists(filepath) == False):
				s.sendto(str('error(filepath do not exist)').encode(),addr)
				continue
			image = load_image(filepath)
			result = model.predict(image)
			acc = np.max(result)
			pos = np.argmax(result)
			category = category_lists[pos]
			s.sendto((str(category)+str(acc)).encode(),addr)
		except BaseException:
			print('error!')
			s.sendto(str('error').encode(),addr)
			continue
		else:
			print('success to mission\n'+'result:'+str(category)+'\n'+'acc'+str(acc))
		
if __name__ == '__main__':
	udpserver()