#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 23:53:06 2021

@author: sania
"""
import torch
import torch.nn as nn
from model.activation import activation_factory
from model.Random_Drops import *

# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, act_type, keep_prob, block_size, 
                 num_point, residual=True, **kwargs):
        super(SpatialGraphConv, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point
        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.A = nn.Parameter(A, requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1
            
        self.act = activation_factory(act_type)   
        self.bn = nn.BatchNorm2d(out_channel)
        
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )
        self.dropS = Randomized_DropBlock_Ske()
        self.dropT = Randomized_DropBlockT_1d(block_size=block_size)
         
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        n, kc, t, v = x.size()
        #x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        #print(self.A.shape)
        #print(x.shape)
        x = torch.einsum('nctv,vw->nctw', (x, self.A * self.edge)).contiguous()
        #x = self.dropS(self.bn(x), self.keep_prob, self.A * self.edge, self.num_point) + self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point)
        x = self.dropT(self.dropS(self.bn(x), self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob) + self.dropT(self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob)
        
        return self.act(x)
    
class SepTemporal_Block(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act_type, edge, A, num_point, keep_prob, block_size, expand_ratio, stride=1, residual=True, **kwargs):
        super(SepTemporal_Block, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point
        padding = (temporal_window_size - 1) // 2
        self.act = activation_factory(act_type)

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = lambda x:0
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )
        self.A = nn.Parameter(A, requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1
        self.dropS = Randomized_DropBlock_Ske()
        self.dropT = Randomized_DropBlockT_1d(block_size=block_size)
        
    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        #x = self.dropT(x, self.keep_prob) + self.dropT(res, self.keep_prob)
        x = self.dropT(self.dropS(x, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob) + self.dropT(self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob)
        return self.act(x)