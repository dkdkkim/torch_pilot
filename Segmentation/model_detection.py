from torch import nn
import torch

def get_conv_layer(in_ch: int,
                   out_ch:  int,
                   kernel_size: int=3,
                   stride: int=1,
                   padding: int=1,
                   bias: bool = True):
    return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def get_up_layer(in_ch: int,
                 out_ch: int,
                 kernel_size: int=2,
                 stride: int=2,
                 up_mode: str='transposed'):
    if up_mode=='transpozed':
        return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

def get_avgpool_layer(kernel_size: int = 2,
                      stride: int = 2,
                      padding: int = 0):
    return nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()

def get_normalization(num_ch):
    return nn.BatchNorm3d(num_ch)

class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()
    def forward(self, layer1, layer2):
        x = torch.cat((layer1, layer2), 1)
        return x

class Transition(nn.Module):
    """
    A helper Moduel that performs a trainsition layer
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 activation: str = 'relu',
                 is_pool: bool = True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation
        self.is_pool = is_pool
        self.norm = get_normalization(self.in_ch)
        self.act = get_activation(self.activation)
        self.conv = get_conv_layer(self.in_ch, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        if self.is_pool:
            self.pool = get_avgpool_layer()

    def forward(self, x):
        conv = self.norm(x)
        conv = self.act(conv)
        conv = self.conv(conv)
        if self.is_pool:
            conv = self.pool(conv)
        return conv

class DenseBlock(nn.Module):
    """
    A helper Module that performs a single denseblock
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 mid_ch: int,
                 activation: str = 'relu',
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mid_ch = mid_ch
        self.activation = activation
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0

        self.conv1 = get_conv_layer(self.in_ch, self.mid_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = get_conv_layer(self.mid_ch, self.out_ch, kernel_size=3, stride=1, padding=self.padding, bias=True)

        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        self.norm1 = get_normalization(self.in_ch)
        self.norm2 = get_normalization(self.mid_ch)

        self.concat = Concatenate()

    def forward(self, x):
        conv = self.norm1(x)
        conv = self.act1(conv)
        conv = self.conv1(conv)

        conv = self.norm2(conv)
        conv = self.act2(conv)
        conv = self.conv2(conv)

        y = self.concat(x, conv)

        return y

class Model(nn.Module):
    def __init__(self,
                 in_ch: int=1,
                 classes: int=2,
                 growth: int=16,
                 initconvstride: int=2,
                 blocksize: list = [6, 12]):
        super().__init__()

        self.blocksize = blocksize
        self.growth = growth
        self.in_ch = in_ch
        self.classes = classes
        self.init_ch = self.growth*2
        self.initconvstride = initconvstride

        ### intial convolution
        self.conv_init=get_conv_layer(self.in_ch, self.init_ch, kernel_size=3, stride=self.initconvstride, padding=1,
                                      bias=True)

        ### dense layers
        self.blocks=[]
        self.trans_layers=[]
        for i in range(len(self.blocksize)):
            if i == 0:
                cur_init_ch = self.init_ch
            cur_blocks=[]
            for j in range(self.blocksize[i]):
                dense_block = DenseBlock(cur_init_ch + j * self.growth, self.growth, self.growth*4)
                cur_blocks.append(dense_block)
            cur_blocks = nn.ModuleList(cur_blocks)
            self.blocks.append(cur_blocks)
            prev_init_ch = cur_init_ch
            cur_init_ch = int((cur_init_ch + self.blocksize[i] * self.growth) / 2)
            trans_layer = Transition(cur_init_ch*2, cur_init_ch, is_pool=False)
            self.trans_layers.append(trans_layer)
        self.trans_layers = nn.ModuleList(self.trans_layers)
        self.blocks = nn.ModuleList(self.blocks)
        ### final convolution
        self.conv_final = get_conv_layer(cur_init_ch, self.classes, kernel_size=1, stride=1, padding=0, bias=True)

        ### initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            method(module.weight, **kwargs)
    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            method(module.bias, **kwargs)

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)
            self.bias_init(module, method_bias, **kwargs_bias)

    def forward(self, x: torch.tensor):

        x = self.conv_init(x)

        for idx, dense_block in enumerate(self.blocks):
            for module in dense_block:
                x = module(x)
            x = self.trans_layers[idx](x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys()
                      if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'








