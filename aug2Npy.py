###
# coding:utf-8
# Author = "liuchang"
# Date:20180606
###


from merge2Augment import *

test_path = 'newtest/'
npy_path = '.'

TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.tfrecords'
PREDICT_SET_NAME = 'predict_set.tfrecords'

INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 512, 512, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 512, 512, 1
TRAIN_SET_SIZE = 2100
VALIDATION_SET_SIZE = 27
TEST_SET_SIZE = 30
PREDICT_SET_SIZE = 30
images_col = 512
images_row = 512

"""
def aug2npy():
	''' convert  augmentated images to npy file'''

	path_train = merge2Augment.aug2train_path	#train path
	path_label = merge2Augment.aug2label_path	#label path

	images_train = os.listdir(path_train)
	images_label = os.listdir(path_label)

	image_train_num = len(images_train)
	image_label_num = len(images_label)

	imgs_train = np.ndarray((image_train_num, 1,, images_row, images_col),type=np.uint8) #generate data (shape , datatyepe) 
	imgs_label = np.ndarray((image_label_num, 1,, images_row, images_col),type=np.uint8) #generate data (shape , datatyepe) 
"""
class  DataProcess(object):
	"""process data . 1 ,generate it to npy file. 2, use npy file to TFRcords"""
	def __init__(self, out_rows, out_cols, aug_path=aug_path, aug_train_path=aug2train_path, 
		aug_label_path=aug2label_path, test_path=test_path, npy_path=npy_path, img_type='tif'):
		super( DataProcess, self).__init__()
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.aug_path = aug_path
		self.aug_train_path = aug2train_path
		self.aug_label_path = aug2label_path
		self.test_path = test_path
		self.npy_path = npy_path
		self.img_type = img_type

	def create_train_data(self):
		'''augment data to generate npy'''
		print 'save train data npy'
		print '*'*30
		count = 0
		i = 0
		''' count train images number , next creat ndarry to save img'''
		for indir in os.listdir(self.aug_path):
			path =  os.path.join(self.aug_path, indir)
			count += len(os.listdir(path))

		imgdatas = np.ndarray((count, self.out_rows, self.out_cols, 1),dtype=np.uint8)
		imglabels = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)

		''' read the image from each images dir to save '''
		for indir in os.listdir(self.aug_path):
			trainpath = os.path.join(self.aug_train_path, indir)
			labelpath = os.path.join(self.aug_label_path, indir)
			print trainpath, labelpath
			''' read the image '''
			imgs = glob.glob(trainpath + '/*.tif')
			for imgname in imgs:
				trainmidname = imgname[imgname.rindex('/')+1:]
				labelmidname = imgname[imgname.rindex('/')+1: imgname.rindex('_')] + '_label.tif'
				print trainmidname, labelmidname
				img = load_img(trainpath + '/' + trainmidname, grayscale=True)
				label = load_img(labelpath + '/' + labelmidname, grayscale=True)
				img = img_to_array(img)
				label = img_to_array(label)
				imgdatas[i] = img 
				imglabels[i] = label
				if 0 == i % 100:
					print 'Done: {0}/{1} images'.format(i, len(imgs))
				i += 1
				print i 
		print 'loading done.', imgdatas.shape
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_label.npy', imglabels)
		print'Saving to .npy files dona.'

	def create_test_data(self):
		'''create test data  to .npy file'''
		print 'create test data .npy file '
		print '*'*30

		count=0
		i=0
		testpath = self.test_path
		
		imgs = glob.glob(testpath +  '/*.tif')
		count = len(imgs)
		testdatas = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		for img in imgs:
			midname = img[img.rindex('/')+1:]
			img = load_img(testpath + '/' + midname, grayscale=True)
			img = img_to_array(img)
			testdatas[i] = img 
			if 0 == i % 100:
				print 'Done {0}/{1} images'.format(i, count)
			i += 1
		print 'loading done', testdatas.shape
		np.save(self.npy_path + '/' + 'imgs_test.npy', testdatas)
		print 'Saving to .npy files done.'

	def load_train_data(self):
		'''read train data including label data, and normolization'''
		
		print 'loading train images...'
		print '*'*30

		imgs_train = np.load(self.npy_path + '/imgs_train.npy')
		imgs_label = np.load(self.npy_path + '/imgs_label.npy')
		imgs_train = imgs_train.astype('float32')
		imgs_label = imgs_label.astype('float32')

		imgs_train /= 255
		mean = imgs_train.mean(axis=0)
		imgs_train -= mean
		imgs_label /= 255
		imgs_label[imgs_label > 0.5] = 1
		imgs_label[imgs_label <= 0.5] = 0
		print 'load train images done.'
		return imgs_train, imgs_label

	def load_test_data(self):
		'''load test train'''
		print 'loading test data...'
		print '*'*30

		imgs_test = np.load(self.npy_path + '/imgs_test.npy')
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis=0) # get each col's mean,
		imgs_test -= mean
		print 'load test data done.'
		return imgs_test

	def show_test_result(self):
		'''show test result'''
		print 'show_test_result'
		print '*'*30

		imgs_test = np.load(self.npy_path + '/imgs_test_label.npy')
		img1 = np.argmax(a=imgs_test[1], axis=-1).astype('uint8')
		cv2.imshow('r', img1 * 100)
		cv2.waitKey(0)

	def write_img_to_tfrecords(self):
		'''convert npy format data to tfrecord format by tensorflow to read'''
		import tensorflow as tf
		from random import shuffle
		import cv2

		print "write_img_to_tfrecords"	
		print '*'*30

		train_set_writer = tf.python_io.TFRecordWriter(TRAIN_SET_NAME)  # will be created
		validation_set_writer = tf.python_io.TFRecordWriter(VALIDATION_SET_NAME)
		test_set_writer = tf.python_io.TFRecordWriter(TEST_SET_NAME)

		aug_image_path = []
		for indir in os.listdir(self.aug_path):
			imgs = glob.glob(os.path.join(self.aug_path, indir) + '/*.tif')
			aug_image_path.extend(imgs)
		print 'Total files are %d '%len(aug_image_path)
		shuffle(aug_image_path) #shuffle the queue

		#get randomed train and label
		for index, image_path in enumerate(aug_image_path[:TRAIN_SET_SIZE]):
			train_image_path = image_path.replace('augment','aug2train')
			train_image_path = train_image_path[:train_image_path.rindex('.tif')] + '_train.tif'
			label_image_path = image_path.replace('augment', 'aug2label')
			label_image_path = label_image_path[:label_image_path.rindex('.tif')] + '_label.tif'

			image = cv2.imread(train_image_path, flags=0)
			label  =cv2.imread(label_image_path, flags=0)

			label[ label <= 100 ] = 0
			label[ label > 100 ] = 1

			example = tf.train.Example(features=tf.train.Features(feature={
				'label' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
				'image_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
				})) # example对象对label和image数据进行封装

			train_set_writer.write(example.SerializeToString()) #  convert serial to string
			if 0 == index%100:
				print 'Done train_set_writing %.2f%%' % (index/TRAIN_SET_SIZE * 100)
		train_set_writer.close()
		print 'Done whole train set writing!'

		#get validation set
		for index, image_path in enumerate(aug_image_path[TRAIN_SET_SIZE:TRAIN_SET_SIZE + VALIDATION_SET_SIZE]):
			validation_image_path = image_path.replace('augment','aug2train')
			validation_image_path = validation_image_path[:validation_image_path.rindex('.tif')] + '_train.tif'
			label_image_path = image_path.replace('augment', 'aug2label')
			label_image_path = label_image_path[:label_image_path.rindex('.tif')] + '_label.tif'

			image = cv2.imread(validation_image_path, flags=0)
			label  =cv2.imread(label_image_path, flags=0)

			label[ label <= 100 ] = 0
			label[ label > 100 ] = 1

			example = tf.train.Example(features=tf.train.Features(feature={
				'label' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
				'image_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
				})) # example对象对label和image数据进行封装

			validation_set_writer.write(example.SerializeToString()) #  convert serial to string
			if 0 == index%100:
				print 'Done train_set_writing %.2f%%' % (index/TRAIN_SET_SIZE * 100)
		validation_set_writer.close()
		print 'Done whole validation set writing!'

		# # test set
		# for index in range(PREDICT_SET_SIZE):
		# 	origin_image_path = ORIGIN_IMAGE_DIRECTORY
		# 	origin_label_path = ORIGIN_LABEL_DIRECTORY
		# 	predict_image = cv2.imread(os.path.join(origin_image_path, '%d.tif' % index), flags=0)
		# 	predict_label = cv2.imread(os.path.join(origin_label_path, '%d.tif' % index), flags=0)
			
		# 	predict_label[predict_label <= 100] = 0
		# 	predict_label[predict_label > 100] = 1
			
		# 	example = tf.train.Example(features=tf.train.Features(feature={
		# 		'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_label.tobytes()])),
		# 		'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_image.tobytes()]))
		# 	}))  # example对象对label和image数据进行封装
		# 	test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# 	if index % 10 == 0:
		# 		print('Done test_set writing %.2f%%' % (index / TEST_SET_SIZE * 100))
		# test_set_writer.close()
		# print("Done whole test_set writing")


if __name__ == '__main__':

	datapro = DataProcess(images_row,images_col)
	# datapro.create_train_data()
	# datapro.create_test_data()
	datapro.write_img_to_tfrecords()
	
