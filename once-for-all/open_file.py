#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   open_file.py
@Time   :   2022/01/08 14:58:13
@Author  :   Rick Xie 
@Version :   1.0
@Contact :   xier2018@mail.sustech.edu.cn
@Desc   :   None
'''


import torch
import numpy as np
import sys
from torchsummary import summary
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import pdb

sys.path.append('/home/rick/nas_rram')
sys.path.append('/home/rick/nas_rram/ofa/once-for-all')
sys.path.append('/home/rick/nas_rram/ofa/once-for-all/ofa/imagenet_classification/elastic_nn/networks')
# sys.path.append('/home/rick/nas_rram/ofa/once-for-all/ofa/imagenet_classification/elastic_nn/networks/ofa_resnets.py')
print(sys.path)

##########################generate_layer_infor##########################
def generate_layer_infor(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(generate_layer_infor)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            generate_layer_infor[m_key] = OrderedDict()
            generate_layer_infor[m_key]["input_shape"] = list(input[0].size())
            generate_layer_infor[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                generate_layer_infor[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                generate_layer_infor[m_key]["output_shape"] = list(output.size())
                generate_layer_infor[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                generate_layer_infor[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            generate_layer_infor[m_key]["nb_params"] = params
            

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    generate_layer_infor = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    # total_params = 0
    # total_output = 0
    # trainable_params = 0
    for layer in generate_layer_infor:
        # input_shape, output_shape, trainable, nb_params
        # line_new = "{:>20}  {:>25} {:>15}".format(
        #     layer,
        #     str(generate_layer_infor[layer]["output_shape"]),
        #     "{0:,}".format(generate_layer_infor[layer]["nb_params"]),
        # )
        # total_params += generate_layer_infor[layer]["nb_params"]
        # total_output += np.prod(generate_layer_infor[layer]["output_shape"])
        # if "trainable" in generate_layer_infor[layer]:
        #     if generate_layer_infor[layer]["trainable"] == True:
        #         trainable_params += generate_layer_infor[layer]["nb_params"]
        # print(line_new)
        # print("layer is ", layer)
        if ((generate_layer_infor[layer]["nb_params"]>0) and ("BatchNorm2d" not in layer)):
            print(layer, generate_layer_infor[layer])

    
#############################################################################
#############################################################################
#############################################################################

# from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAProxylessNASNets, OFAResNets, OFAResNets18
# from OFAResNets18 import sample_active_subnet, sample_active_subnet
# ofa_model = torch.load('/home/rick/nas_rram/ofa_data/exp/normal2kernel/checkpoint/model_best.pth.tar')  
# print(ofa_model.keys())    

# state_dict = ofa_model['state_dict']
# print(state_dict.keys()) 
# ofa_weights = state_dict['blocks.0.conv.depth_conv.conv.weight']
# print(ofa_weights)

# neurosim_model = torch.load('/home/rick/nas_rram/neurosim_log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/latest.pth')  
# print(neurosim_model.keys())
# neurosim_weights = neurosim_model['module.features.0.weight']
# print(neurosim_weights)

# ofa_specialized = torch.load('/home/rick/nas_rram/ofa_data/.torch/ofa_specialized/pixel1_lat@143ms_top1@80.1_finetune@75/init')  
# print(ofa_specialized.keys())
# state_dict = ofa_specialized['state_dict']
# # np.savetxt('/home/rick/nas_rram/ofa_data/tmp/ofa_specialized_state_dict',str(state_dict))
# print(state_dict)

# model = torch.load('/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/resnet18-5c106cde.pth')
# del model['fc.bias']
# model1 = model.copy()
# print(model1.keys())
# torch.save(model1, '/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/resnet18_without_fb.pth')

# vgg8_model = torch.load('/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/resnet18_without_fb.pth')

# model = torch.load('/home/rick/nas_rram/ofa_data/exp_resnet/kernel_depth2kernel_depth_width/phase2/checkpoint/intact_model_best.pth.tar')
model = torch.load('/home/rick/nas_rram/ofa_data/exp_resnet/normal2kernel/checkpoint/intact_model_best.pth.tar')
teacher = torch.load('/home/rick/nas_rram/ofa_data/exp_resnet/teachernet/checkpoint/intact_model_best.pth.tar')
# model = torch.load('/home/rick/nas_rram/ofa_data/exp_resnet/kernel_depth2kernel_depth_width/phase2/checkpoint/intact_model_best.pth.tar')

# print(str(vgg8_model.parameters()))
# model.sample_active_subnet()

# subnet = model.get_active_subnet(preserve_weight=False)
# subnet_config = model.get_active_net_config()

# subnet = model.sample_active_subnet()
# print(subnet)
# print(list(model.children()))
print("----------------------------------------------------------------")
# print(model)
# print(subnet_config)
# print(teacher)


# torch.save({'state_dict': checkpoint['state_dict']}, best_path, _use_new_zipfile_serialization=False)
# print('test best')
# save intact model
# torch.save(subnet, '/home/rick/nas_rram/ofa_data/sample_subnet/sample_resnet18/intact_subnet_best.pth.tar',_use_new_zipfile_serialization=False)

subnet = torch.load('/home/rick/nas_rram/ofa_data/sample_subnet/sample_resnet18/intact_subnet_best.pth.tar')

# import pdb
# pdb.set_trace()
# summary(teacher,(3,32,32))
# print(model)
generate_layer_infor(subnet,(3,32,32))

# for i,j in vgg8_model.named_parameters():
#     # if 'conv2' in i:
#     #     print(i,j)
#     print(i)

# print(str(vgg8_model))


# densenet40_model = torch.load('/home/rick/nas_rram/DNN_NeuroSim_V1.3/Inference_pytorch/log/DenseNet40.pth')
# print(densenet40_model.keys())



