import torch
import torch.nn as nn


class _DenseLayer(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate):
        super(_DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout2d(drop_rate)
        )
   
    def forward(self, x):
        out = self.conv(x)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, growth_rate=16, num_layer=4, drop_rate=0.5):
        super(DenseBlock, self).__init__()
        for i in range(num_layer-1):
            layer = _DenseLayer(
                in_ch=in_ch + i*growth_rate,
                out_ch=growth_rate,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i+1), layer)
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch + int(growth_rate*(num_layer-1)),
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        dimension = [x]
        for name, layer in self.named_children():
            new_dimension = layer(torch.cat(dimension, 1))
            dimension.append(new_dimension)
        #out = self.out(torch.cat(dimension, 1))
        return new_dimension
      


class ResBlock(nn.Module):
    def __init__(self, channals_out, channals_in):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channals_in, channals_in*2, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channals_in*2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channals_in*2, channals_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channals_out)
        self.relu2 = nn.ReLU()
        self.cnn_skip = nn.Conv2d(channals_in, channals_out, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        if x.size(1) != residual.size(1):
            x = self.cnn_skip(x)
        x = x + residual
        x = self.relu2(x)
        return x


class down(nn.Module):
    def __init__(self, dropout, in_ch, out_ch, kernel_size=4, padding=1, stride=2):
        super(down, self).__init__()
        self.same = nn.Sequential(
            DenseBlock(out_ch=out_ch, in_ch=in_ch),
            nn.Dropout2d(dropout)
        )
        self.d = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        x_skip = self.same(x)
        down = self.d(x_skip)
        return x_skip, down


class up(nn.Module):
    def __init__(self, dropout, in_ch, in_m_ch, out_ch, kernel_size=4, padding=1, stride=2):
        super(up, self).__init__()
        self.u = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(in_ch),
            nn.ReLU()
        )
        self.h = nn.Sequential(
            DenseBlock(out_ch=out_ch, in_ch=in_m_ch),
            nn.Dropout2d(dropout)
        )

    def forward(self, x, x_skip):
        x_up = self.u(x)
        #print(x.size())
        x = torch.cat([x_up, x_skip], 1)
        #print(x.size())
        x = self.h(x)
        return x


class up_top(nn.Module):
    def __init__(self, dropout, in_ch,in_m_ch, out_ch, kernel_size=4, stride=2, padding=1):
        super(up_top, self).__init__()
        self.u = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.Tanh()
        )
        self.h = nn.Sequential(
            nn.Conv2d(in_m_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((28, 28)),
            nn.Sigmoid(),
            nn.Dropout2d(dropout)
        )

    def forward(self, x, x_skip):
        x_up = self.u(x)
        x = torch.cat([x_up, x_skip], 1)
        x = self.h(x)
        return x


class top_out(nn.Module):
    def __init__(self, dropout, in_ch, kernel_size=3, stride=1, padding=1):
        super(top_out, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 1, kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.LayerNorm((256, 256)),
            nn.Sigmoid(),
        )
  
    def forward(self, x):
        return self.cnn(x)


class sampling(nn.Module):
    def __init__(self, dropout=0.5, order=2):
        super(sampling, self).__init__()
        self.down = nn.Sequential()
        for i in range(order):
            self.down.add_module('pooling' + str(i+1), nn.AvgPool2d(2))
        self.up = nn.Upsample(scale_factor=2**order, mode='nearest')

    def forward(self, x):
        out = self.down(x.view(-1, 1, 1024, 1024))
        #out = self.up(out)
        return out


class unet(nn.Module):
    def __init__(self, dropout=0.5):
        super(unet, self).__init__()
        self.s1_down = down(dropout, in_ch=1, out_ch=64, kernel_size=4, padding=1, stride=2)#64x256x256
        self.s2_down = down(dropout, in_ch=64, out_ch=128, kernel_size=4, padding=1, stride=2)#128x128x128
        self.s3_down = down(dropout, in_ch=128, out_ch=256, kernel_size=4, padding=1, stride=2)#256x64x64
        self.s4_down = down(dropout, in_ch=256, out_ch=512, kernel_size=4, padding=1, stride=2)#512x32x32
        self.s5_down = down(dropout, in_ch=512, out_ch=1024, kernel_size=4, padding=1, stride=2)#1024x16x16
        self.s4_up = up(dropout, in_ch=1024, in_m_ch=1024+512, out_ch=256)
        self.s3_up = up(dropout, in_ch=256, in_m_ch=512, out_ch=128)
        self.s2_up = up(dropout, in_ch=128, in_m_ch=256, out_ch=64)
        self.s1_up = up(dropout, in_ch=64, in_m_ch=128, out_ch=32)
        self.output = top_out(dropout, in_ch=32)
        self._initialize_weights()
 

    def forward(self, speckles):
        x_s1_skip, x_s1_down = self.s1_down(speckles.view(-1, 1, 256, 256))
        x_s2_skip, x_s2_down = self.s2_down(x_s1_down)
        x_s3_skip, x_s3_down = self.s3_down(x_s2_down)
        x_s4_skip, x_s4_down = self.s4_down(x_s3_down)
        x_bottom, _ = self.s5_down(x_s4_down) #16x16

        x_s4_up = self.s4_up(x_bottom, x_s4_skip)
        x_s3_up = self.s3_up(x_s4_up, x_s3_skip)
        x_s2_up = self.s2_up(x_s3_up, x_s2_skip)
        x_s1_up = self.s1_up(x_s2_up, x_s1_skip)
        out = self.output(x_s1_up)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        
