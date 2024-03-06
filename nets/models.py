from nets.resnet_utils import BasicBlock, Bottleneck, conv3x3, conv1x1
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, output_dim=64, n_classes=10, width=1):
        super(SimpleCNN, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*width, kernel_size=3, bias=False) 
        self.conv1_bn = nn.BatchNorm2d(16*width)
        self.conv2 = nn.Conv2d(16*width, 16*width, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(16*width)        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(16*width, 32*width, 3, stride=2, bias=False) 
        self.conv3_bn = nn.BatchNorm2d(32*width)
        
        self.conv4 = nn.Conv2d(32*width, 32*width, 3, bias=False)
        self.conv4_bn = nn.BatchNorm2d(32*width)
        self.conv5 = nn.Conv2d(32*width, 32*width, 1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(32*width)
        
        self.conv6 = nn.Conv2d(32*width, 64*width, 3, stride=2, bias=False) 
        self.conv6_bn = nn.BatchNorm2d(64*width)        

        self.conv7 = nn.Conv2d(64*width, 64*width, 3, bias=False)
        self.conv7_bn = nn.BatchNorm2d(64*width)
        
        self.conv8 = nn.Conv2d(64*width, 64*width, 1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(64*width)        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(64*width, output_dim)
        self.fc1_bn = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.fc2_bn = nn.BatchNorm1d(output_dim)
        self.fc3 = nn.Linear(output_dim, n_classes)

        


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = self.relu(x)
        
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = self.relu(x)        


        x = self.avgpool(x)

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu(x)        

        y = self.fc3(x)        

        return 0, x, y, 0, 0
    
    
    
    
    

class ResNet_18(nn.Module):
    def __init__(self, args,  num_class=2):
        super(ResNet_18, self).__init__()
        
        
        norm_layer = nn.BatchNorm2d
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
        )
        #self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        
        downsample_l1 = nn.Sequential(conv1x1(in_planes=64, out_planes=64, stride=2), norm_layer(64))
        self.layer1 = nn.Sequential(
            BasicBlock(inplanes=64, planes=64, stride=2, downsample=downsample_l1, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),      
            
            BasicBlock(inplanes=64, planes=64, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
               

        )      
        
        downsample_l2 = nn.Sequential(conv1x1(in_planes=64, out_planes=128, stride=2), norm_layer(128))
        self.layer2 = nn.Sequential(
            
            BasicBlock(inplanes=64, planes=128, stride=2, downsample=downsample_l2, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            BasicBlock(inplanes=128, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer)        

        )            

        
        downsample_l3 = nn.Sequential(conv1x1(in_planes=128, out_planes=256, stride=2), norm_layer(256))
        self.layer3 = nn.Sequential(
            
            BasicBlock(inplanes=128, planes=256, stride=2, downsample=downsample_l3, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            BasicBlock(inplanes=256, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                          
        )                 

        
        downsample_l4 = nn.Sequential(conv1x1(in_planes=256, out_planes=512, stride=2), norm_layer(512))
        self.layer4 = nn.Sequential(
            
            BasicBlock(inplanes=256, planes=512, stride=2, downsample=downsample_l4, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            BasicBlock(inplanes=512, planes=512, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            

        )         
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        num_ftrs = args.out_dim
        self.fc1 = nn.Linear(512, num_ftrs)
        self.fc1_bn = nn.BatchNorm1d(num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2_bn = nn.BatchNorm1d(num_ftrs)
        self.fc3 = nn.Linear(num_ftrs, num_class)


        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1 = self.initial_conv(x)
        #x = self.max_pool(out1)
        
        out2 = self.layer1(out1)
        x = self.layer2(out2)
        x = self.layer3(x)
        x = self.layer4(x)

                
        x = self.avgpool(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        preds = self.fc3(x)
        
        return x, x, preds, out1, out2
    
    
    
class ResNet_50(nn.Module):
    def __init__(self, args,  num_class=2):
        super(ResNet_50, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        #self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        
        downsample_l1 = nn.Sequential(conv1x1(in_planes=64, out_planes=256, stride=1), norm_layer(256))
        self.layer1 = nn.Sequential(
            
            Bottleneck(inplanes=64, planes=64, stride=1, downsample=downsample_l1, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                        

        )      
        
        downsample_l2 = nn.Sequential(conv1x1(in_planes=256, out_planes=512, stride=2), norm_layer(512))
        self.layer2 = nn.Sequential(
            
            Bottleneck(inplanes=256, planes=128, stride=2, downsample=downsample_l2, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),          

        )            

        
        downsample_l3 = nn.Sequential(conv1x1(in_planes=512, out_planes=1024, stride=2), norm_layer(1024))
        self.layer3 = nn.Sequential(
            
            Bottleneck(inplanes=512, planes=256, stride=2, downsample=downsample_l3, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),   
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                

        )                 

        
        downsample_l4 = nn.Sequential(conv1x1(in_planes=1024, out_planes=2048, stride=2), norm_layer(2048))
        self.layer4 = nn.Sequential(
            
            Bottleneck(inplanes=1024, planes=512, stride=2, downsample=downsample_l4, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),

        )             
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        ### Projection head
        num_ftrs = args.out_dim
        self.fc1 = nn.Linear(2048, num_ftrs)
        self.fc1_bn = nn.BatchNorm1d(num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2_bn = nn.BatchNorm1d(num_ftrs)
        self.fc3 = nn.Linear(num_ftrs, num_class)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1 = self.initial_conv(x)
        #x = self.max_pool(out1)
        
        out2 = self.layer1(out1)
        x = self.layer2(out2)
        x = self.layer3(x)
        x = self.layer4(x)

                
        x = self.avgpool(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        preds = self.fc3(x)
        
        return x, x, preds, out1, out2