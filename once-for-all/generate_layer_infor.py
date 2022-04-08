#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   generate_layer_infor.py
@Time   :   2022/04/01 13:13:37
@Author  :   Rick Xie 
@Version :   1.0
@Contact :   xier2018@mail.sustech.edu.cn
@Desc   :   None
'''


import torch
import sys
import torch.nn as nn
from collections import OrderedDict
import pdb
import numpy as np
import csv
from torchsummary import summary


sys.path.append('/home/rick/nas_rram')
sys.path.append('/home/rick/nas_rram/ofa/once-for-all')
sys.path.append('/home/rick/nas_rram/ofa/once-for-all/ofa/imagenet_classification/elastic_nn/networks')
torch.nn.Module.dump_patches = True
##########################generate_layer_infor##########################
def generate_layer_infor(model, input_size, batch_size=-1, device="cuda"):
    """Basic Description:
    This function generates the layer information for neurosim input in csv format.
    
    Description:
    
    
    Args:
    model: Intact model
    input_size: Image size as input with channel number
    batch_size: Batch size (default: 1)
    device: Device on (default:"cuda", option: "cuda", "cpu")
    
    Returns:
    No return value but with a generated csv file.
    
    """
    

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
            
            if hasattr(module, "kernel_size"):
                generate_layer_infor[m_key]["kernel_size"] = module.kernel_size
                # print(module.kernel_size)
            

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

    layer_infor = OrderedDict()
    count = 0
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

            input_shape = generate_layer_infor[layer]["input_shape"][1:]
            output_shape = generate_layer_infor[layer]["output_shape"][1:]
        
            
            input_shape = input_shape[::-1]
            output_shape = output_shape[::-1]
            
            if("kernel_size" in generate_layer_infor[layer]):
                kernel_size = generate_layer_infor[layer]["kernel_size"]
                output_shape = [kernel_size[0],kernel_size[1], output_shape[2]]

            
            if ((len(input_shape) == 1) and (len(output_shape)) == 1):
                input_shape = [1, 1, input_shape[0]]
                output_shape = [1, 1, output_shape[0]]
            
            layer_infor[count] = input_shape + output_shape
            
            count += 1
            # print(generate_layer_infor[layer]["kernel_size"])
            
            
            
    count = 0
    stride = 0
    pooling = 0
    for count in range(0, len(layer_infor)):
        if count <len(layer_infor)-1:
            if (layer_infor[count][0] == 2*layer_infor[count+1][0]):
                stride = 2
                pooling = 0
            elif (layer_infor[count][0] == 4*layer_infor[count+1][0]):
                stride = 2
                pooling = 1
            elif ((layer_infor[count][0] > layer_infor[count+1][0]) and
                 (layer_infor[count][0] != 2*layer_infor[count+1][0]) and
                 (layer_infor[count][0] != 4*layer_infor[count+1][0])):
                stride = 1
                pooling = 1
            else:
                stride = 1
                pooling = 0
        else:
            stride = 1
            pooling = 0
            
        layer_infor[count].insert(len(layer_infor[count]), pooling)
        layer_infor[count].insert(len(layer_infor[count]), stride)
        
        with open('./DNN_NeuroSim_V1.3/Inference_pytorch/NeuroSIM/NetWork.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in layer_infor.items():
                writer.writerow(v)
                
    
    # print(layer_infor)
    

model = torch.load('/home/rick/nas_rram/ofa_data/exp_resnet/normal2kernel/checkpoint/intact_model_best.pth.tar')
teacher = torch.load('/home/rick/nas_rram/ofa_data/exp_resnet/teachernet/checkpoint/intact_model_best.pth.tar')

# demo = torch.load('/home/rick/nas_rram/NeuroSim_modified/Inference_pytorch/log/OFA_ResNet18/intact_model_best.pth.tar')
demo = torch.load('/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/intact_resnet18_without_fb.pth')

subnet = torch.load('/home/rick/nas_rram/ofa_data/sample_subnet/sample_resnet18/intact_subnet_best.pth.tar')

final_searched_result = torch.load('/home/rick/nas_rram/ofa_data/exp_resnet_multi_width/kernel_depth2kernel_depth_width/phase2/checkpoint/intact_model_best.pth.tar')
final_searched_result_subnet = final_searched_result.get_active_subnet()
torch.save(final_searched_result_subnet,'/home/rick/nas_rram/ofa_data/neurosim_input/intact_model_best.pth.tar')
# generate_layer_infor(teacher,(3,32,32))
# generate_layer_infor(subnet,(3,32,32))
generate_layer_infor(final_searched_result_subnet,(3,32,32))
# print(demo)




