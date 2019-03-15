import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function

use_binary = True

class BinaryLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
#         print(input.shape, weight.shape)
        weight_mask = None
        if use_binary:
            weight_mask = (weight > 1) | (weight < -1)
            weight = torch.sign(weight)
        #weight_b = torch.sign(weight)
        #weight_b = weight
        ctx.save_for_backward(input, weight, weight_mask, bias)

        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
#         print(grad_output.shape)
        input, weight, weight_mask, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-1, -2).matmul(input)
            if weight_mask is not None:
                grad_weight.masked_fill_(weight_mask, 0.0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias

class BinaryStraightThroughFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        grad_input = grad_output.clone()
        mask = (grad_input > 1) | (grad_input < -1)
        grad_input = grad_input.masked_fill_(mask, 0.0)
        return grad_input

binary_linear = BinaryLinearFunction.apply
bst = BinaryStraightThroughFunction.apply

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (math.sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.bias is not None:
            return binary_linear(input, self.weight, self.bias)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class BinaryStraightThrough(nn.Module):
    def __init__(self, inplace=False):
        super(BinaryStraightThrough, self).__init__()

    def forward(self, input):
        return bst(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
