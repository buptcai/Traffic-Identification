from keras.models import Model
from keras.layers import GlobalAveragePooling2D,Dense
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class_num = 15
size = 224
checkpoint_filepath = './checkpoint_again/model-ep{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}-val_loss{val_loss:.3f}.h5'
tensorboard_filepath  = './checkpoint_again'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config = config)
KTF.set_session(session)

base_model = InceptionV3(weights = None,include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation = 'relu')(x)

predictions = Dense(class_num,activation = 'softmax')(x)
model = Model(inputs = base_model.input,outputs = predictions)
for i,layer in enumerate(base_model.layers):
	print(i,layer.name)

for layer in base_model.layers[:101]:
	layer.trainable = False
for layer in base_model.layers[101:]:
	layer.trainable = True

model.compile(optimizer = SGD(lr = 0.00001,momentum = 0.9),loss = 'categorical_crossentropy',metrics = ["accuracy"])

model.load_weights('./checkpoint/model-ep029-loss0.035-val_acc0.814-val_loss0.808.h5',by_name = False)

train_datagen = ImageDataGenerator(rescale = 1./255)

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

checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',period = 1)#every epoch output a result
tensorboard = TensorBoard(log_dir = tensorboard_filepath,write_images= True)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

model.fit_generator(
	train_generator,
	steps_per_epoch = 100,
	epochs = 1000,
	validation_data = validation_generator,
	validation_steps = 20,
	callbacks = [checkpoint,tensorboard,early_stopping],

	)