from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

#设置路径为环境变量，否则plot_model函数执行会报错
os.environ["PATH"]+= os.pathsep+'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class_num = 15
size = 224
checkpoint_filepath = './checkpoint/model-ep{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}.h5'
tensorboard_filepath  = './checkpoint'

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config = config)
KTF.set_session(session)

#Inceptionv3模型，加载ImageNet的与训练权重，不保留顶层的三个全连接层
base_model = InceptionV3(weights = 'imagenet',include_top = False)
#print(base_model.summary())
#plot_model(base_model,to_file = 'InceptionV3.png')

#Add一个空域全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)
#Add两个全连接层
x = Dense(1024,activation = 'relu')(x)
predictions = Dense(class_num,activation = 'softmax')(x)

#合并层，构建一个新模型
model = Model(inputs = base_model.input,outputs = predictions)
#print(model.summary())
#plot_model(model,to_file = 'InceptionV3_2.png')

#得到各层的信息
#for layer in model.layers:
#	print(layer.get_config())

for layer in base_model.layers:
	layer.trainable = False
#编译模型
model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ["accuracy"]) #rmsprop默认参数

#训练数据生成
train_datagen = ImageDataGenerator(
	rescale = 1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(
	'./dataset/train',
	target_size = (size,size),
	batch_size = 16,
	class_mode = 'categorical')

#测试数据生成
test_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = test_datagen.flow_from_directory(
	'./dataset/test',
	target_size = (size,size),
	batch_size = 16,
	class_mode = 'categorical')

model.fit_generator(
	train_generator,
	steps_per_epoch = 20,
	epochs = 1,
	validation_data = validation_generator, #every epoch
	validation_steps = 20,
	verbose = 1)

#冻结网络的前部分，训练剩余的inception blocks
for i,layer in enumerate(base_model.layers):
	print(i,layer.name)

for layer in base_model.layers[:249]:
	layer.trainable = False
for layer in base_model.layers[249:]:
	layer.trainable = True

model.compile(optimizer = SGD(lr = 0.0001,momentum = 0.9),loss = 'categorical_crossentropy',metrics = ["accuracy"])

checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',period = 1)
tensorboard = TensorBoard(log_dir = tensorboard_filepath,write_images= True)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

model.fit_generator(
	train_generator,
	steps_per_epoch = 200,
	epochs = 200,
	validation_data = validation_generator,
	validation_steps = 20,
	callbacks = [checkpoint,tensorboard,early_stopping]
	)