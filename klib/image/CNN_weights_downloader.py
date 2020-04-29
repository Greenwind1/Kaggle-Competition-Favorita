# -*- coding: utf-8 -*-

from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile

xception = Xception()
vgg16 = VGG16()
vgg19 = VGG19()
res50 = ResNet50()
inception3 = InceptionV3()
inception_res2 = InceptionResNetV2()
mobile = MobileNet()
dense121 = DenseNet121()
dense169 = DenseNet169()
dense201 = DenseNet201()
nasnet_l = NASNetLarge()
nasnet_m = NASNetMobile()
