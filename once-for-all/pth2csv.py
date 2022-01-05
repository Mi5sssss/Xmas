import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str, default='/home/rick/nas_rram/ofa_data/exp/normal2kernel/checkpoint/model_best.pth.tar')
parser.add_argument('-t','--target',type=str,default='/home/rick/nas_rram/ofa_data/layer_record')
args = parser.parse_args()

        
def pth2csv(pth_path = args.path,target_path = args.target):
    '''This is function help to change pth to csv.

    Args:
        pth_path (str): The path of .pth format file.
        target_path (str): The path where you want to save .csv files.

    Returns:
        BufferedFileStorage: A buffered writable file descriptor
    '''
    ofa_model = torch.load(pth_path)
    state_dict = ofa_model['state_dict']
    
    for i in state_dict:
    # print(i+':'+str(state_dict[i].data.size()))
    
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
    
    print('Transform done.')      
        
            
            
if __name__ == '__main__':
    pth2csv(args.path,args.target)
    
    
    
        
    
        
