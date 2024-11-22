import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_dims(no_maxpool, nr_conv, nr_filters):
            input_dim = 64
            for _ in range(nr_conv): #convolution is padded so does not alter input size
                if not no_maxpool: #assuming always kernel size 2x2, and stride 2
                    input_dim /= 2

            return int(input_dim * input_dim * nr_filters[-1])


class CNN(nn.Module):
    """modelo com: 
    up to 6 convolution (nr filters optional) + relu + maxpooling (optional) layers , 
    arbitrary fully connected + relu layers, w/ dropout (between 1st and 2nd) optional 
    softmax (instead of relu) on last layer
    rule:
    for a convolution layer w/out padding, if input is NxN, output will be  MxM, where:
        M = (N − F )/S + 1, where F=kernel_size, S=stride
    """
    def __init__(self, dropout_prob, no_maxpool=False, nr_conv=0, nr_filters=[], nr_fc=1, fc_size=[62]):
        
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        self.conv_layers = nn.ModuleList()
        self.fc_layers  = nn.ModuleList()

        self.nr_conv = nr_conv
        self.nr_fc = nr_fc

        #convolutional layers
        channels = 3
        for i in range(nr_conv):
            conv_layer = nn.Conv2d(in_channels= channels, out_channels=nr_filters[i], kernel_size=(3,3), stride=1, padding='same')
            channels = nr_filters[i]
            self.conv_layers.append(conv_layer)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_prob)

        if not no_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        #fully connected layers
        in_features = calc_dims(no_maxpool, nr_conv, nr_filters)
        for i in range(nr_fc):
            out_features = fc_size[i] #fc_size must always end w/ 62

            if i == nr_fc -1: #last layer
                fc_layer = nn.Linear(in_features=in_features, out_features=out_features)
            else:
                fc_layer = nn.Linear(in_features=in_features, out_features= out_features)

            in_features = fc_size[i]
            self.fc_layers.append(fc_layer)

        

                
    def forward(self, input): #input will be of form (Batch size, 3, 64, 64)
        
        # conv and relu layers
        for conv in self.conv_layers:
            input = conv(input)
            input = self.relu(input)
            # max-pool layer if using it
            if not self.no_maxpool:
                input = self.maxpool(input)
        

        # flatten to fit fully connected layer
        input = input.reshape(((input.shape[0], input.shape[1] * input.shape[2] * input.shape[3]))) 

        for i, fc in enumerate(self.fc_layers):
            input = fc(input)
            if i < self.nr_fc - 1: #don't do relu on last layer
                input = self.relu(input)

            if i == 0 and self.nr_fc > 1: #dropout from 1st to 2nd layer
                input = self.drop(input)
        
        return F.log_softmax(input,dim=1)