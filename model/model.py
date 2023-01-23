# cd /home/uniwa/students3/students/22905553/linux/phd_codes/action_recognition/
import sys
sys.path.insert(0, '')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import *
from model.adjGraph import adjGraph
from utils import import_class

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x
    
class embed(nn.Module):
    def __init__(self, dim, dim1, att_type, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(                        
                norm_data(dim),
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        #self.attention =  Attention_Layer(dim1,  att_type=att_type)

    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        return x#self.attention(x)


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 max_frame,
                 graph,
                 act_type, 
                 bias,
                 edge,
                 block_size):
        super(Model, self).__init__()
        
        self.num_class =  num_class
        temporal_window_size = 3
        max_graph_distance = 2
        keep_prob = 0.9
        Graph = import_class(graph)
        A_binary = torch.Tensor(Graph().A_binary)
        #A = torch.rand(3,25,25).cuda()#.to(num_class.dtype).to(num_class.device)
        #self.graph_hop = adjGraph(**graph_args)
        #A = torch.tensor(self.graph_hop.A, dtype=torch.float32, requires_grad=False)
        #self.register_buffer('A', A)
        
        # channels
        D_embed = 64
        c1 = D_embed*2
        c2 = c1 * 2     
        #c3 = c2 * 2    
       
        
        
        self.joint_embed = embed(2, D_embed, att_type='stja', norm=True, bias=bias)
        #self.dif_embed = embed(2, D_embed, att_type='stja', norm=True, bias=bias) #601
        #self.attention =  Attention_Layer(D_embed,  max_frame, act_type, att_type='stja')
        
        self.sgcn1 = SpatialGraphConv(D_embed, c1, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True)
        self.tcn11 = SepTemporal_Block(c1, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True)
        self.tcn12 = SepTemporal_Block(c1, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True)
        
        self.sgcn2 = SpatialGraphConv(c1, c2, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True)
        self.tcn21 = SepTemporal_Block(c2, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True)
        self.tcn22 = SepTemporal_Block(c2, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True)
        
        
        self.fc = nn.Linear(c2, num_class)
        #init_param(self.modules())
    
    def forward(self, x):        
        
        N, C, T, V = x.size()
        #dy = x
        
        # Dynamic Representation        
        pos = x.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        #print(pos.shape)
        #dif = pos[:, :, :, 1:] - pos[:, :, :, 0:-1] #  
        #dif = torch.cat([dif.new(N, dif.size(1), V, 1).zero_(), dif], dim=-1)
        
        pos = self.joint_embed(pos)        
        #dif = self.dif_embed(dif)
        dy = pos #+ dif
        #dy = dif
        dy = dy.permute(0,1,3,2).contiguous() # N, C, T, V   
        #print(dy.shape)
        #dy = self.attention(dy)
        #dy.register_hook(lambda g: print(g))
      
        #########################
        out = self.tcn12(self.tcn11(self.sgcn1(dy)))
        out = self.tcn22(self.tcn21(self.sgcn2(out)))
        #print(out.shape)
        out_channels = out.size(1)
        out = out.reshape(N, out_channels, -1)   
        #print(out.shape)
        out = out.mean(2)
        #print(out.shape)
        out = self.fc(out)
        
        return out
        

if __name__ == "__main__":
    # For debugging purposes
#     cd /home/uniwa/students3/students/22905553/linux/phd_codes/Light_Fall
    import sys
    sys.path.append('..')
    import thop
    from thop import clever_format
     
    model = Model(
        num_class=2,
        num_point=25,
        max_frame=300,
        graph='graph.ntu_rgb_d.AdjMatrixGraph',
        act_type = 'relu',
        bias = True,
        edge = True,
        block_size=41
    )
    #model = model.cuda()
    macs, params = thop.profile(model, inputs=(torch.randn(1,2,300,25),), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    #N, C, T, V, M = 6, 3, 300, 25, 2
    
    x = torch.randn(1, 2, 300, 25)#.cuda()   
    out = model.forward(x)
    
    ##
    # Drop frame
    import torch.nn.functional as F
    keep_prob = 0.9
    block_size = 41
    input = torch.randn(1,2,300,25)
    n,c,t,v = input.size()
    input1 = input.permute(0,1,3,2).contiguous().view(1,c*v,t)
    input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1)
    gamma = (1. - keep_prob) / block_size
    M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)
    Msum = F.max_pool1d(M, kernel_size=[block_size], stride=1, padding=block_size // 2)
    
    mask = (1 - Msum)
    drop = (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)
    
    idx = torch.randperm(Msum.shape[2])
    a = Msum[idx].view(Msum.size())
    
    idx = torch.randperm(Msum.shape[2])
    a = Msum[:,:,idx].view(Msum.size())
    mask = (1 - a)
    drop = (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)
    
    # Drop joints
    from utils import import_class
    input = torch.randn(1,2,300,25)
    n,c,t,v = input.size()
    Graph = import_class('graph.ntu_rgb_d.AdjMatrixGraph')
    A_binary = torch.Tensor(Graph().A_binary)
    input_abs = torch.mean(torch.mean(torch.abs(input), dim=2), dim=1)
    input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
    gamma = (1. - keep_prob) / (1 + 1.92)
    M_seed = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0))
    M = torch.matmul(M_seed, A_binary)
    M[M > 0.001] = 1.0
    M[M < 0.5] = 0.0
    mask = (1 - M).view(n, 1, 1, 25)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
