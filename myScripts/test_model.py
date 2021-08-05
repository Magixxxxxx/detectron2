import torch
from torchvision.models import resnet
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import sys,os

sys.path.append(".")

def testMask(path):
    from collections import defaultdict
    mask_density = defaultdict(list)
    mask_mean = defaultdict(list)

    if path.endswith('.pth'):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        for name,p in state_dict['model'].items():
            if 'mask' in name:
                density = torch.sum(p.view(-1)<0.005).numpy() / len(p.view(-1))
                mean = torch.mean(p.view(-1))
                max_mask = torch.max(p.view(-1))
                min_mask = torch.min(p.view(-1))
                print("{} \n zeroed:{:.3f} mean:{:.3f} max:{:.3f} min:{:.3f}".format(name,density,mean,max_mask,min_mask))
    else:
        for model_pth in os.listdir(path):
            state_dict = torch.load(os.path.join(path,model_pth), map_location=torch.device('cpu'))
            for name,p in state_dict['model'].items():
                if 'mask' in name:
                    density = torch.sum(p.view(-1)>0.005).numpy() / len(p.view(-1))
                    mean = torch.mean(p.view(-1))
                    mask_density[name].append(density)
                    mask_mean[name].append(mean)

    # np.save('mask_density.npy',mask_density)
    # np.save('mask_mean.npy',mask_mean)

def analyseParams(model, layer):
    state_dict = torch.load(model, map_location=torch.device('cpu'))
    for name,p in state_dict['model'].items():
        if layer in name:
            print(p.view(-1))

def compareModels(model1, model2, o_f):
    state_dict1 = torch.load(model1, map_location=torch.device('cpu'))
    state_dict2 = torch.load(model2, map_location=torch.device('cpu'))
    params1 = np.array([])
    params2 = np.array([])

    pre_l = 'stem'
    
    for (name1,p1),(name2,p2) in zip(state_dict1['model'].items(),state_dict2['model'].items()):
        if 'backbone' in name1:
            cur_l = name1.split('.')[1]
            if 'mask' in name1 or name1 == 'proposal_generator.rpn_head.conv.weight':

                if cur_l == pre_l:
                    active1 = np.array(p1.view(-1)>0.005, bool)
                    active2 = np.array(p2.view(-1)>0.005, bool)
                    params1 = np.hstack([params1,active1])
                    params2 = np.hstack([params2,active2])
                    continue

                common = np.sum([a and b for a,b in zip(params1, params2)])
                s_of_1 = np.sum(params1)
                s_of_2 = np.sum(params2)
                s_of_12 = np.sum(params1) + np.sum(params2) - common
                s_of_layer = len(params1)
                print(active1, s_of_1, s_of_2, s_of_12, s_of_layer)

                com_by_1 = np.round(common / s_of_1, 2)
                com_by_2 = np.round(common / s_of_2, 2)       
                com_by_12 = np.round(common / s_of_12, 2)       
                com_by_all = np.round(common / s_of_layer, 2)
                print("{} c_by_1: {} c_by_2: {} c_by_12: {} c_by_all: {} \n".format(cur_l, com_by_1, com_by_2, com_by_12, com_by_all))
                o_f.write("{} c_by_1: {} c_by_2: {} c_by_12: {} c_by_all: {} \n".format(cur_l, com_by_1, com_by_2, com_by_12, com_by_all))
                
                pre_l = cur_l
                params1 = np.array([])
                params2 = np.array([])
                continue


def compareLayers(model):
    state_dict = torch.load(model, map_location=torch.device('cpu'))
    for name,p in state_dict['model'].items():
        if 'backbone.body.conv1.mask_real' == name:
            pre_active = p.view(-1)>0.005
            continue
        if 'mask' in name:
            pre_active = active
            active = p.view(-1)>0.005
            print(name, sum(active == pre_active).numpy()/len(p.view(-1)))
            # print(sum(p1.view(-1)>0.005==p.view(-1)>0.005), len(p1.view(-1)))


if __name__ == '__main__':
    # testMask("output/pb[body]voc[5,5]/model_0001999.pth")

    with open('compare2.out', 'w') as o_f:
        for i in range(1, 10):
            for j in range(i+1, 10):
                print("compare {} {}".format(i, j))
                o_f.write("compare {} {}\n".format(i, j))
                model1 = "output/pb[body]voc[{},{}]/model_0002999.pth".format(i,i)
                model2 = "output/pb[body]voc[{},{}]/model_0002999.pth".format(j,j)
                if j == 8: continue
                compareModels(model1, model2, o_f)

