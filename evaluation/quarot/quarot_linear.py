import logging
from tkinter import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quarot_utils import random_hadamard_matrix

from .quant_utils import WeightQuantizer, ActivationQuantizer


class VGGTQuantizedLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear):
        super(VGGTQuantizedLinear, self).__init__()

        self.args = args
        self.not_rot = args.not_rot
        self.not_smooth = args.not_smooth
        self.lwc = args.lwc
        self.lac = args.lac
        self.rv = args.rv
        self.linear = linear
        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)

        self.act_quantizer = ActivationQuantizer(bits=args.a_bits, sym=not(args.a_asym), lac=self.lac,
                                                 groupsize=args.a_groupsize, )
        if self.lwc:
            lwc_dim = self.linear.weight.shape[0] if self.lwc else -1
            init_value = 4.
            self.clip_factor_w_max = nn.Parameter(torch.ones((lwc_dim, 1)) * init_value, requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(torch.ones((lwc_dim, 1)) * init_value, requires_grad=True)
            self.sigmoid = nn.Sigmoid()

        if not args.not_smooth:
            self.act_quantizer.register_buffer("act_scale", None)  
            self.register_parameter("channel_wise_scale",
                                nn.Parameter(torch.ones((1, self.linear.weight.shape[1])))) 
            self.smooth_quant_momentum = 0.95
            self.smooth_quant_alpha = 0.5
            self.smooth_quant_running_stat = True 

        if not self.not_rot:
            self.register_parameter("rotation_matrix",
                                  torch.nn.Parameter(random_hadamard_matrix(self.linear.weight.shape[1], "cuda").to(dtype=torch.float32)))

        self.ori_mode = True
        self.train_mode = False
        self.eval_mode = False


    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight
    

    def _ori_forward(self, hidden_states):
        weight = self.linear.weight.data
        bias = self.linear.bias
        if not self.not_smooth :
            if  self.smooth_quant_running_stat:
                if not self.not_rot:
                    hidden_states = torch.matmul(hidden_states.float(), self.rotation_matrix)
                    weight = torch.matmul(weight.float(), self.rotation_matrix)
               
                cur_act_scale = hidden_states.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros(1).to(hidden_states)
                if self.act_quantizer.act_scale.abs().mean() == 0:
                    self.act_quantizer.act_scale = cur_act_scale
                else:
                    self.act_quantizer.act_scale = self.act_quantizer.act_scale * self.smooth_quant_momentum + cur_act_scale * (
                            1 - self.smooth_quant_momentum)
            else:
                assert self.act_quantizer.act_scale is not None
                assert self.act_quantizer.act_scale.mean() != 0

        return F.linear(hidden_states, weight, bias)

    def _train_forward(self, hidden_states):
        weight = self.linear.weight.data
        if not self.not_rot:
            weight = torch.matmul(weight, self.rotation_matrix)
        if not self.not_smooth:
            weight = weight * self.channel_wise_scale
        if self.lwc:
            weight = self.apply_wclip(weight)

        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)

        if not self.not_rot:
            hidden_states = torch.matmul(hidden_states, self.rotation_matrix)
        if not self.not_smooth:
            hidden_states = hidden_states / self.channel_wise_scale

        hidden_states = self.act_quantizer(hidden_states)
        bias = self.linear.bias
        output = F.linear(hidden_states, weight, bias)

        return output

    def _eval_forward(self, hidden_states):
        x_dtype = hidden_states.dtype
        if not self.not_rot:
            hidden_states = torch.matmul(hidden_states.float(), self.rotation_matrix)
        if not self.not_smooth:
            hidden_states = hidden_states / self.channel_wise_scale
        hidden_states = self.act_quantizer(hidden_states).to(x_dtype)
        output = self.linear(hidden_states)
        return output

    def forward(self, hidden_states):
        if self.ori_mode:
            return self._ori_forward(hidden_states)
        if self.train_mode:
            return self._train_forward(hidden_states) 
        if self.eval_mode:
            return self._eval_forward(hidden_states)

    def reparameterize(self):
        target_device = self.linear.weight.device
        ori_dtype = self.linear.weight.dtype
        weight = self.linear.weight.data.detach().to(torch.float32) 
        if not self.not_rot:
            weight = torch.matmul(weight, self.rotation_matrix.to(weight.device))        
        if not self.not_smooth:
            weight = weight * self.channel_wise_scale.to(weight.device)

        if self.lwc:
            weight = self.apply_wclip(weight)

        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)
        self.linear.weight.data = weight.to(device=target_device, dtype=ori_dtype)

        self.ori_mode = False
        self.train_mode = False
        self.eval_mode = True
