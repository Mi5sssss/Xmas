#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   pth2csv.py
@Time   :   2022/01/08 11:17:56
@Author  :   Rick Xie 
@Version :   1.0
@Contact :   xier2018@mail.sustech.edu.cn
@Desc   :   None
'''


import torch
import numpy as np
import argparse
import os
import collections

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str, default='/home/rick/nas_rram/ofa_data/exp/normal2kernel/checkpoint/model_best.pth.tar')
parser.add_argument('-t','--target',type=str,default='/home/rick/nas_rram/ofa_data/layer_record')


        
def pth2csv(pth_path = '/home/rick/nas_rram/ofa_data/exp/normal2kernel/checkpoint/model_best.pth.tar',
            target_path = '/home/rick/nas_rram/ofa_data/layer_record'):
    '''This is function help to change pth to csv.

    Args:
        pth_path (str): The path of .pth format file.
        target_path (str): The path where you want to save .csv files.

    Returns:
        None
    '''
    ofa_model = torch.load(pth_path)
    state_dict = ofa_model['state_dict']
    
    for i in state_dict:
        print(i+':'+str(state_dict[i].data.size()))
        
        # for mobilenet
        if 'inverted_bottleneck.conv.conv.weight' in i: 
            # blocks.1.mobile_inverted_conv.inverted_bottleneck.conv.conv.weight : torch.Size([96, 16, 1, 1])
            folder_name = '/inverted_bottleneck.conv.conv.weight'
            # print(i,':',state_dict[i].data.size())
            data_reshaped = state_dict[i].data.reshape(state_dict[i].data.size(0),
                                                       state_dict[i].data.size(1)).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
        elif 'depth_conv.conv.conv.weight' in i:
            folder_name = '/depth_conv.conv.conv.weight'
            # blocks.1.conv.depth_conv.conv.conv.weight:torch.Size([64, 1, 3, 3])
            data_reshaped = state_dict[i].data.reshape(state_dict[i].data.size(0),
                                                       -1).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
        # for resnet
        elif 'conv1.conv.weight' in i:
            folder_name = '/conv1.conv.weight'
            # print(i,':',state_dict[i].data.size())
            data_reshaped = state_dict[i].data.reshape(state_dict[i].data.size(0),
                                                       state_dict[i].data.size(1)).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
        elif 'conv2.conv.weight' in i:
            folder_name = '/conv2.conv.weight'
            # blocks.1.conv.depth_conv.conv.conv.weight:torch.Size([64, 1, 3, 3])
            data_reshaped = state_dict[i].data.reshape(state_dict[i].data.size(0),
                                                       -1).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
        elif 'classifier.linear.weight' in i:
            folder_name = '/classifier.linear.weight'
            data_reshaped = state_dict[i].data.reshape(state_dict[i].data.size(0),
                                                       -1).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
        elif 'input_stem.0.conv.weight' in i:
            folder_name = '/input_stem.0.conv.weight'
            data_reshaped = state_dict[i].data.reshape(state_dict[i].data.size(0),
                                                       -1).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
    
    
    print('Transform done.')  


def pth2csv_official(pth_path = '/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/resnet18-5c106cde.pth',
            target_path = '/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official'):
    '''This is function help to change pth to csv of pytorch official model.

    Args:
        pth_path (str): The path of .pth format file.
        target_path (str): The path where you want to save .csv files.

    Returns:
        None
    '''
    
    
    ofa_model = torch.load(pth_path)
    
    for i in ofa_model:
        # print(i+':'+str(ofa_model[i].data.size()))
        
        if 'downsample.0.weight' in i: 
            folder_name = '/downsample.0.weight'
            data_reshaped = ofa_model[i].data.reshape(ofa_model[i].data.size(0),
                                                       ofa_model[i].data.size(1)).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
            
        elif 'conv1.weight' in i:
            folder_name = '/conv1.weight'

            data_reshaped = ofa_model[i].data.reshape(ofa_model[i].data.size(0),
                                                       -1).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
        elif 'conv2.weight' in i:
            folder_name = '/conv2.weight'
            data_reshaped = ofa_model[i].data.reshape(ofa_model[i].data.size(0),
                                                       -1).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
        if 'fc.weight' in i: 
            folder_name = '/fc.weight'
            data_reshaped = ofa_model[i].data.reshape(ofa_model[i].data.size(0),
                                                       ofa_model[i].data.size(1)).cpu().numpy()
            if not os.path.exists(target_path+folder_name):
                os.mkdir(target_path+folder_name)
            np.savetxt(target_path+folder_name+'/'+i+'.csv',
                       data_reshaped,delimiter=',',fmt='%.5f')
            
    
    
    print('Transform done.')  
        
def key_transform(ofa_path='/home/rick/nas_rram/ofa_data/exp_resnet/teachernet/checkpoint/model_best.pth.tar',
                  template_path='/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/resnet18-5c106cde.pth',
                  target_path='/home/rick/nas_rram/ofa_data/modified_model/resnet_18'):
    '''This is function help to change the keys in ofa trained model to the neurosim input.

    Args:
        ofa_path (str): The path of .pth format file.
        template_path (str): The template path of the target.
        target_path (str): The path where you want to save the neurosim-like .pth files.

    Returns:
        None
    '''
    ofa_model = torch.load(ofa_path)
    template_model = torch.load(template_path)
    state_dict = ofa_model['state_dict']
    
    target_model = collections.OrderedDict.fromkeys(template_model.keys()) # start a target dictionary
    # print(template_model.keys())
    tem_target_model = collections.OrderedDict.fromkeys(()) # start a target dictionary
    # target_model = dict.fromkeys(()) # start a target dictionary
    tem = collections.OrderedDict.fromkeys(())
    

    # for i,j in zip(state_dict, template_model):
    #     print(i,'+++',j)
        # if not 'num_batches_tracked' in i:
        #     # print(i+':'+str(state_dict[i]))
        #     tem = {j:state_dict[i]}
        #     tem_target_model.update(tem)
        #     tem.clear()
        #     print(j+':'+str(tem_target_model[j].data.size()))
    
    # torch.save(target_model, target_path+'/target_model.pth',
    #            _use_new_zipfile_serialization=False)
    
    for i in state_dict:
        if not 'num_batches_tracked' in i:
            
            # input layer
            if 'input_stem.0.' in i:
                if 'conv.weight' in i:
                    tem = {'conv1.weight':state_dict[i]}
                    tem_target_model.update(tem)
                    tem.clear()
                if 'bn.' in i:
                    if 'weight' in i:
                        tem = {'bn1.weight':state_dict[i]}
                        tem_target_model.update(tem)
                        tem.clear()
                    if 'bias' in i:  
                        tem = {'bn1.bias':state_dict[i]}
                        tem_target_model.update(tem)
                        tem.clear()
                    if 'running_mean' in i:
                        tem = {'bn1.running_mean':state_dict[i]}
                        tem_target_model.update(tem)
                        tem.clear()
                    if 'running_var' in i:
                        tem = {'bn1.running_var':state_dict[i]}
                        tem_target_model.update(tem)
                        tem.clear()      
            # fc layer            
            elif 'classifier.linear.' in i:
                if 'weight' in i:
                    tem = {'fc.weight':state_dict[i]}
                    tem_target_model.update(tem)
                    tem.clear()
                if 'bias' in i:
                    tem = {'fc.bias':state_dict[i]}
                    tem_target_model.update(tem)
                    tem.clear()
                    
            # block layer
            else:
            # in layerA.B.convC // BlockN, ConvC // N = 4A+2B+C-5 
                if 'blocks.' in i:
                    if (i[i.index('blocks.')+len('blocks.')+1] == '.'):
                        N = int(i[i.index('blocks.')+len('blocks.')])
                    elif (i[i.index('blocks.')+len('blocks.')+1] != '.'):
                        N = int(i[i.index('blocks.')+len('blocks.'):i.index('blocks.')+len('blocks.')+2])
                if 'conv' in i:
                    if (i[i.index('conv')+len('conv')] != '.'):
                        C = int(i[i.index('conv')+len('conv')])
                # print(N,C)
                
                for B in [0,1]:
                    if (N+5-C-2*B)%4 == 0:
                        A = int((N+5-C-2*B)/4)
                        break
                # print(A,B,C,N)
                
                if 'conv'+str(C) in i:
                    if 'conv.weight' in i:
                        tem = {'layer'+str(A)+'.'+str(B)+'.'+
                               'conv'+str(C)+'.weight':state_dict[i]}
                        tem_target_model.update(tem)
                        tem.clear()
                    if 'bn.' in i:
                        if 'weight' in i:
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                   'bn'+str(C)+'.weight':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear()
                        if 'bias' in i:  
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                   'bn'+str(C)+'.bias':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear()
                        if 'running_mean' in i:
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                   'bn'+str(C)+'.running_mean':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear()
                        if 'running_var' in i:
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                   'bn'+str(C)+'.running_var':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear() 
                            # tem = {'layer'+str(A)+'.'+str(B)+'.'+str()
                            #        :state_dict[i]}
                            # tem_target_model.update(tem)
                            # tem.clear()
                            
                if 'downsample.' in i:
                    if 'conv.weight' in i:
                        tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                'downsample.0'+'.weight':state_dict[i]}
                        tem_target_model.update(tem)
                        tem.clear()
                    if 'bn.' in i:
                        if 'weight' in i:
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                    'downsample.1'+'.weight':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear()
                        if 'bias' in i:  
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                    'downsample.1'+'.bias':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear()
                        if 'running_mean' in i:
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                    'downsample.1'+'.running_mean':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear()
                        if 'running_var' in i:
                            tem = {'layer'+str(A)+'.'+str(B)+'.'+
                                    'downsample.1'+'.running_var':state_dict[i]}
                            tem_target_model.update(tem)
                            tem.clear() 
                    

                    
                    
        # print(i+':'+str(tem_target_model[i].data.size()))
    print(tem_target_model.keys())
    torch.save(tem_target_model, target_path+'/target_model.pth',
               _use_new_zipfile_serialization=False)
        






            
if __name__ == '__main__':
    args = parser.parse_args()
    
    key_transform()
    # pth2csv(args.path,args.target)
    # pth2csv('/home/rick/nas_rram/ofa_data/exp/teachernet/checkpoint/model_best.pth.tar',
            # '/home/rick/nas_rram/ofa_data/layer_record/teacher')
    
    # pth2csv('/home/rick/nas_rram/ofa_data/exp_resnet/teachernet/checkpoint/model_best.pth.tar',
    #         '/home/rick/nas_rram/ofa_data/layer_record_resnet18/teachernet')
    
    # pth2csv_official('/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/resnet18-5c106cde.pth',
    #     '/home/rick/nas_rram/ofa_data/neurosim_model/resnet_official/resnet18-5c106cde')
    
    
    
        
    
        
