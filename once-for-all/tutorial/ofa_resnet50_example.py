#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   ofa_resnet50_example.py
@Time   :   2022/04/25 16:24:28
@Author  :   Rick Xie 
@Version :   1.0
@Contact :   xier2018@mail.sustech.edu.cn
@Desc   :   None
'''
import sys
sys.path.append('./once-for-all')
# build ofa resnet50
from ofa.model_zoo import ofa_net
ofa_network = ofa_net('ofa_resnet50', pretrained=False)
# accuracy predictor
print(ofa_network)
import torch
from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder
from ofa.utils import download_url

image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
arch_encoder = ResNetArchEncoder(
	image_size_list=image_size_list, depth_list=ofa_network.depth_list, expand_list=ofa_network.expand_ratio_list,
    width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST
)

acc_predictor_checkpoint_path = download_url(
    'https://hanlab.mit.edu/files/OnceForAll/tutorial/ofa_resnet50_acc_predictor.pth',
    model_dir='~/.ofa/',
)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# acc_predictor = AccuracyPredictor(arch_encoder, 400, 3,
#                                   checkpoint_path=acc_predictor_checkpoint_path, device=device)
acc_predictor = AccuracyPredictor(arch_encoder, 400, 3,
                                  checkpoint_path=None, device=device)

print('The accuracy predictor is ready!')
print(acc_predictor)

# build efficiency predictor
from ofa.nas.efficiency_predictor import ResNet50FLOPsModel

efficiency_predictor = ResNet50FLOPsModel(ofa_network)

# search
import random

for i in range(10):
    subnet_config = ofa_network.sample_active_subnet()
    image_size = random.choice(image_size_list)
    subnet_config.update({'image_size': image_size})
    predicted_acc = acc_predictor.predict_acc([subnet_config])
    predicted_efficiency = efficiency_predictor.get_efficiency(subnet_config)

    print(i, '\t', predicted_acc, '\t', '%.1fM MACs' % predicted_efficiency)
