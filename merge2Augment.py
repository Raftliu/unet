# coding:utf-8
# Author='liuchang'
# time:'20180530'

from  keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import split_data
import os
import numpy as np
import glob
import cv2

merge_path = "merge/"
aug_path = 'augment/'
train_path = 'newtrain/'
label_path = 'newlabel/'

aug2train_path = 'aug2train/'
aug2label_path = 'aug2label/'

class Augment(object):
    
    '''use it to aug these image and label.'''
    '''use label as a channel of image '''

    def  __init__(self, train_path=train_path, label_path=label_path, merge_path=merge_path, aug_path=aug_path, aug2train_path=aug2train_path, aug2label_path=aug2label_path, img_type='tif' ):
        ''' usage: init all varible'''
        self.train_imgs = glob.glob(train_path + '/*.' + img_type) #train set
        self.label_imgs = glob.glob(label_path + '/*.'+ img_type) #label set
        self.train_path=train_path
        self.label_path=label_path
        self.merge_path=merge_path
        self.aug_path=aug_path
        self.aug2train_path=aug2train_path
        self.aug2label_path=aug2label_path
        self.img_type=img_type
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
                rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')
    def augmentation(self):
        '''读入3通道的train和label, 分别转换成矩阵, 然后将label的第一个通道放在train的第2个通处, 做数据增强 '''
        print 'runing program augmentation!'
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug = self.aug_path
        print('%d images \n%d labels' % (len(trains), len(labels)))
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0
        if not os.path.lexists(path_merge):
            os.mkdir(path_merge)
        if not os.path.lexists(path_aug):
            os.mkdir(path_aug)
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)  # 读入train
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)  # 读入label
            x_t = img_to_array(img_t)                                    # 转换成矩阵
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]                                  # 把label当做train的第三个通道
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)      # 保存合并后的图像
            img = x_t
            img = img.reshape((1,) + img.shape)                          # 改变shape(1, 512, 512, 3)
            savedir = path_aug + "/" + str(i)                      # 存储合并增强后的图像
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            print("running %d doAugmenttaion" % i)
            self.do_augmentate(img, savedir, str(i))                      # 数据增强

    def do_augmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=70):
        # augmentate one image
        datagen = self.datagen
        i = 0
        for _ in datagen.flow(
                img,
                batch_size=batch_size,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format):
            i += 1
            if i > imgnum:
                break
       
    def split_merge(self):
        # 读入合并增强之后的数据(aug_merge), 对其进行分离, 分别保存至 aug_train, aug_label
        print("running splitMerge")

        # split merged image apart
        path_merge = self.aug_path       # 合并增强之后的图像
        path_train = self.aug2train_path       # 增强之后分离出来的train
        path_label = self.aug2label_path       # 增强之后分离出来的label

        if not os.path.lexists(path_train):
            os.mkdir(path_train)
        if not os.path.lexists(path_label):
            os.mkdir(path_label)

        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            print(path)
            train_imgs = glob.glob(path + "/*." + self.img_type)  # 所有训练图像
            savedir = path_train + "/" + str(i)                   # 保存训练集的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + str(i)                   # 保存label的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:         # rindex("/") 是返回'/'在字符串中最后一次出现的索引
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]  # 获得文件名(不包含后缀)
                # midname = os.path.split(imgname)[1]
                # midname = os.path.splitext(imgname)[1]
                # print imgname , '###' ,midname
                # print os.path.splitext(imgname)[0]
                # print os.path.splitext(imgname)[1]
                # print os.path.split(imgname)[0]
                # print os.path.split(imgname)[1]
                # print "*" *30

                img = cv2.imread(imgname)      # 读入训练图像
                img_train = img[:, :, 2]  # 训练集是第2个通道, label是第0个通道
                img_label = img[:, :, 0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train)  # 保存训练图像和label
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label" + "." + self.img_type, img_label)


if __name__ == '__main__':
    '''test program'''
    print 'do augmentation'
    aug = Augment()
 #   aug.augmentation() 
    aug.split_merge()
