import torch
import torch.nn as nn
from models.miniVIT import VIT
import torch.nn.functional as F

class BasicBlock3d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, size, stride=1, downsample=None, conv_num=3):
        super(BasicBlock3d, self).__init__()
        self.layers = []
        self.same_padding = size//2
        for i in range(conv_num):
            if i == 0:
                self.layers.append(nn.Conv3d(inplanes, planes, kernel_size=(3,3,size), stride=stride, padding=(1,1,self.same_padding), bias=False))
                self.layers.append(nn.BatchNorm3d(planes))
                self.layers.append(nn.LeakyReLU(0.1))
            else:
                self.layers.append(nn.Conv3d(planes, planes, kernel_size=(3,3,size), stride=(1,1,1), padding=(1,1,self.same_padding), bias=False))
                self.layers.append(nn.BatchNorm3d(planes))
                self.layers.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*self.layers)
        self.seblock = SEblock3D(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.layers(x)
        out = self.seblock(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class BottleNeck3d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, size, stride=[2,1], downsample=None, conv_num=3):
        super(BottleNeck3d, self).__init__()
        self.same_padding = size//2
        self.layers1 = []
        for i in range(conv_num):
            if i == 0:
                self.layers1.append(nn.Conv3d(inplanes, planes, kernel_size=size, stride=(1,1, stride[0]), padding=self.same_padding, bias=False))
                self.layers1.append(nn.BatchNorm3d(planes))
                self.layers1.append(nn.LeakyReLU(0.1))
            else:
                self.layers1.append(nn.Conv3d(planes, planes, kernel_size=size, stride=1, padding=self.same_padding, bias=False))
                self.layers1.append(nn.BatchNorm3d(planes))
                self.layers1.append(nn.LeakyReLU(0.1))
        self.layers1 = nn.Sequential(*self.layers1)

        self.layers2 = []
        for i in range(conv_num):
            if i == 0:
                self.layers2.append(nn.Conv3d(planes, planes, kernel_size=size, stride = stride[1], padding=self.same_padding, bias=False))
                self.layers2.append(nn.BatchNorm3d(planes))
                self.layers2.append(nn.LeakyReLU(0.1))
            else:
                self.layers2.append(nn.Conv3d(planes, planes, kernel_size=size, stride=1, padding=self.same_padding, bias=False))
                self.layers2.append(nn.BatchNorm3d(planes))
                self.layers2.append(nn.LeakyReLU(0.1))
        self.layers2 = nn.Sequential(*self.layers2)

        self.neck = []
        self.neck.append(nn.Conv3d(inplanes, planes, kernel_size=size, stride=(1,1, stride[0]), padding=self.same_padding, bias=False))
        self.neck.append(nn.BatchNorm3d(planes))
        self.neck.append(nn.LeakyReLU(0.1))
        self.neck = nn.Sequential(*self.neck)

        self.seblock = SEblock3D(planes)
        self.downsample = downsample

    def forward(self, x):                   
        neck = self.neck(x)
        out = self.layers1(x)
        out = self.seblock(out)
        out += neck

        residual = out
        out = self.layers2(out)
        out = self.seblock(out)
        out += residual

        return out

class SEblock3D(nn.Module):
    def __init__(self, inplanes):
        super(SEblock3D, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(inplanes, int(inplanes / 4), kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(int(inplanes / 4), inplanes, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class backbone(nn.Module):
    def __init__(self, input_channel=1, hidden=48):
        super(backbone, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv3d(input_channel, hidden, kernel_size=(1,1,15), stride=(1,1,1), padding=(0, 0, 7), bias=False)
        self.bn1 = nn.BatchNorm3d(hidden)
        self.ks = [3, 3, 5, 7]

        # block1
        self.inplanes = hidden
        self.layers1 = nn.Sequential()
        self.layers1.add_module('layer_1', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 1), size=self.ks[0], conv_num=1))

        # block2
        self.layers2_1 = nn.Sequential()
        self.layers2_1.add_module('layer_2_1_1', self._make_layer3d(BottleNeck3d, self.inplanes, 1, stride=(2, 1), size=self.ks[1], conv_num=2))
        self.layers2_1.add_module('layer_2_1_2', self._make_layer3d(BottleNeck3d, self.inplanes, 1, stride=(2, 1), size=self.ks[1], conv_num=2))

        self.layers2_2 = nn.Sequential()
        self.layers2_2.add_module('layer_2_2_1', self._make_layer3d(BottleNeck3d, self.inplanes, 1, stride=(2, 1), size=self.ks[2], conv_num=2))
        self.layers2_2.add_module('layer_2_2_2', self._make_layer3d(BottleNeck3d, self.inplanes, 1, stride=(2, 1), size=self.ks[2], conv_num=2))

        self.layers2_3 = nn.Sequential()
        self.layers2_3.add_module('layer_2_3_1', self._make_layer3d(BottleNeck3d, self.inplanes, 1, stride=(2, 1), size=self.ks[3], conv_num=2))
        self.layers2_3.add_module('layer_2_3_2', self._make_layer3d(BottleNeck3d, self.inplanes, 1, stride=(2, 1), size=self.ks[3], conv_num=2))
        
        self.trans_layers = VIT(n_embd=hidden*3, n_layer=6, n_head=4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden*3, 6, 6)) #  b, seq_len, c, w, h 
        
        self.is_ischemia = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        # SPECT
        self.ischemia_intensity = MLP(hidden*3*2, hidden*3, 1, 3)  # regress
        self.ischemia_area = MLP(hidden*3*2, hidden*3, 1, 3) # regress
        self.xin_jian = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        self.qian_bi = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        self.jiange_bi = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        self.xia_bi = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        self.ce_bi = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        # CTA
        self.LAD = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        self.LCX = MLP(hidden*3*2, hidden*3, 1, 3) # bce
        self.RCA = MLP(hidden*3*2, hidden*3, 1, 3) # bce
               

    def _make_layer3d(self, block, planes, blocks, stride=(1, 1, 2), size=15, conv_num=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, size, stride, downsample, conv_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, qrs, t):
        # qrs
        x0 = qrs.clone()
        x0 = self.relu(self.bn1(self.conv1(x0)))
        x0 = self.layers1(x0)
        x2_1 = self.layers2_1(x0)
        x2_2 = self.layers2_2(x0)
        x2_3 = self.layers2_3(x0)
        
        out_qrs = torch.cat([x2_1, x2_2, x2_3], dim=1) 
        b,c,w,h,seq_len = out_qrs.shape
        out_qrs = out_qrs.permute(0,4,1,2,3)  #  b, seq_len, c, w, h 
        cls_tokens_qrs = self.cls_token.expand(b, -1, -1, -1, -1)
        feat_qrs = torch.cat((cls_tokens_qrs, out_qrs), dim = 1)
        feat_qrs = self.trans_layers(feat_qrs) # 2, 26, 166, 6, 6
        cls_head_qrs = feat_qrs[:,0,:,:,:] #2, 144, 6, 6]
        avg_cls_head_qrs = F.adaptive_max_pool2d(output_size=1, input=cls_head_qrs).squeeze()
        
        # t
        x0 = t.clone()
        x0 = self.relu(self.bn1(self.conv1(x0)))
        x0 = self.layers1(x0)
        x2_1 = self.layers2_1(x0)
        x2_2 = self.layers2_2(x0)
        x2_3 = self.layers2_3(x0)
        
        out_t = torch.cat([x2_1, x2_2, x2_3], dim=1) 
        b,c,w,h,seq_len = out_t.shape
        out_t = out_t.permute(0,4,1,2,3)  #  b, seq_len, c, w, h 
        cls_tokens_t = self.cls_token.expand(b, -1, -1, -1, -1)
        feat_t = torch.cat((cls_tokens_t, out_t), dim = 1)
        feat_t = self.trans_layers(feat_t) # 2, 26, 166, 6, 6
        cls_head_t = feat_t[:,0,:,:,:] #2, 144, 6, 6]
        avg_cls_head_t = F.adaptive_max_pool2d(output_size=1, input=cls_head_t).squeeze()
        
        # fusion
        avg_cls_head = torch.cat((avg_cls_head_qrs, avg_cls_head_t), dim=-1)
        
        # prediction head
        is_ischemia = self.is_ischemia(avg_cls_head)
        # SPECT
        ischemia_intensity = self.ischemia_intensity(avg_cls_head)
        ischemia_area = self.ischemia_area(avg_cls_head)
        xin_jian = self.xin_jian(avg_cls_head)
        qian_bi = self.qian_bi(avg_cls_head)
        jiange_bi = self.jiange_bi(avg_cls_head)
        xia_bi = self.xia_bi(avg_cls_head)
        ce_bi = self.ce_bi(avg_cls_head)
        # CTA
        LAD = self.LAD(avg_cls_head)
        LCX = self.LCX(avg_cls_head)
        RCA = self.RCA(avg_cls_head)        

        output_spect = {'ischemia_intensity':ischemia_intensity, 'ischemia_area':ischemia_area,
                        'xin_jian':xin_jian, 'qian_bi':qian_bi,'jiange_bi': jiange_bi, 'xia_bi':xia_bi, 'ce_bi':ce_bi}        
        output_cta = {'LAD':LAD, 'LCX':LCX, 'RCA':RCA}
        ouput_dict = {'is_ischemia':is_ischemia, 'output_spect':output_spect, 'output_cta':output_cta}
                
        return ouput_dict


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def test():
    net = backbone()
    x = torch.randn([8, 1, 6, 6, 100])
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    net(x, x)  # 1, 1152, 150

# torch.Size([2, 1, 6, 6, 600]) torch.Size([2]) torch.Size([2, 1, 5, 600]) torch.Size([2, 4])
# 30,000,000   5,144,974
# test()

















