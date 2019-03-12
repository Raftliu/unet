#coding:utf-8
#Author:liuchang
#Data:20180614
from keras.callbacks import ModelCheckpoint
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
from aug2Npy import *

class myunet(object):
	"""build unet to train data . get model"""
	def __init__(self, img_row=images_row, img_col=images_col):
		self.img_row = images_row
		self.img_col = images_col

	def load_train_data(self):
		mydata = DataProcess(self.img_row, self.img_col)
		imgs_train ,imgs_label= mydata.load_train_data()
		imgs_label = to_categorical(imgs_label, num_classes=2)
		return imgs_train, imgs_label

	def load_test_data(self):
		mydata = DataProcess(self.img_row, self.img_col)
		imgs_test = mydata.load_test_data()
		return imgs_test

	def get_unet(self):
		inputs = Input((self.img_row, self.img_col, 1))

		conv1_1 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
		conv1_2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1_1)
		pool1 = MaxPooling2D(pool_size=2)(conv1_2)

		conv2_1 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
		conv2_2 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2_1)
		pool2 = MaxPooling2D(pool_size=2)(conv2_2)

		conv3_1 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool2)
		conv3_2 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3_1)
		pool3 = MaxPooling2D(pool_size=2)(conv3_2)

		conv4_1 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool3)
		conv4_2 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4_1)
		drop4 = Dropout(0.5)(conv4_2)
		pool4 = MaxPooling2D(pool_size=2)(drop4)

		conv5_1 = Conv2D(1024, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool4)
		conv5_2 = Conv2D(1024, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5_1)
		drop5 = Dropout(0.5)(conv5_2)

		#encode
		upconv6 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
		merge6 = concatenate([conv4_2, upconv6], axis=3)
		conv6_1 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge6)
		conv6_2 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer= 'he_normal')(conv6_1)

		upconv7 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6_2))
		merge7 = concatenate([conv3_2, upconv7], axis=3)
		conv7_1 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge7)
		conv7_2 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer= 'he_normal')(conv7_1)

		upconv8 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_2))
		merge8 = concatenate([conv2_2, upconv8], axis=3)
		conv8_1 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge8)
		conv8_2 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer= 'he_normal')(conv8_1)

		upconv9 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_2))
		merge9 = concatenate([conv1_2, upconv9], axis=3)
		conv9_1 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge9)
		conv9_2 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer= 'he_normal')(conv9_1)

		conv10 = Conv2D(2, 1, padding='same', activation='softmax', kernel_initializer='he_normal')(conv9_2)
		# conv10 = Softmax()(conv10)

		model = Model(input=inputs, output=conv10)

		model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
		print 'model compile!'
		return model

	def train(self):
		print 'loading data'
		imgs_train, imgs_label = self.load_train_data()
		print 'loading data done'
		model = self.get_unet()
		print 'get unet'

		#save model and data
		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
		print 'Fitting model...'
		model.fit(x=imgs_train, y=imgs_label, validation_split=0.2, batch_size=1, epochs=1, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])

	def test(self):
		print 'loading data'
		imgs_test = self.load_test_data()
		print 'loading data done'
		model = self.get_unet
		model.load_weights('unet.hdf5')
		print 'predict test data'

if __name__ == '__main__':
	unet = myunet()
	unet.get_unet()
	unet.train()
