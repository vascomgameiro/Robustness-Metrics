import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class CNN(nn.Module):
    """
    Constructor receives: layer objects and num_classes. can be conv layer or fc layer. 
    Each is defined below and has multiple options for architecture.
    """

    def __init__(self, conv_layers, fc_layers, num_classes):
        
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList(conv_layers.layers)
        self.fc_layers = nn.ModuleList(fc_layers.layers)
        self.num_classes = num_classes
        self.name = conv_layers.name + fc_layers.name
                
    def forward(self, input): #input will be of form (Batch size, 3, 64, 64)
        
        # conv layers
        for layer in self.conv_layers:
            input = layer(input)
        
        # flatten to fit fully connected layer
        input = input.reshape(((input.shape[0], input.shape[1] * input.shape[2] * input.shape[3]))) 

        #fully connected layers
        for layer in self.fc_layers:
            input = layer(input)
            
        return input
    
        

class Conv():
     
     def __init__(self, nr_conv=0, nr_filters=[], maxpool_batchnorm =True):
        """
        nr_conv -> nr of convolution layers
        nr_filters -> list with filters for each layer
        maxpool_batchnorm -> if true, both batch normalization and maxpooling are exacuted after each convolution
        """
        self.name = f"conv{nr_conv}{maxpool_batchnorm}"
        self.layers = nn.ModuleList()

        input_size = 64
        channels = 3 #rgb

        for i in range(nr_conv):
            #convolution does not alter dimensions
            conv_layer = nn.Conv2d(in_channels= channels, out_channels=nr_filters[i], kernel_size=(3,3), stride=1, padding='same')

            channels = nr_filters[i]
            self.layers.append(conv_layer)

            self.name += f"_{channels}"

            if maxpool_batchnorm:
                batchnorm_layer = nn.BatchNorm2d(channels)
                self.layers.append(batchnorm_layer)

            self.layers.append(nn.ReLU())

            if maxpool_batchnorm:
                maxpool_layer = nn.MaxPool2d(kernel_size=(2,2), stride=2)
                self.layers.append(maxpool_layer)
                input_size /= 2 #maxpooling halves dimensions
        
        self.finaldim = int(input_size*input_size*channels)

class FC():
     
     def __init__(self, nr_fc = 1, fc_size = [62], act_funs = [], dropouts = [], in_features = 0):
        """
        nr_fc -> nr of fully connected layers
        fc_size -> list with nr of nodes for each layer
        act_funs -> list with activation function name (needs to correspond to pytorch name) for each layer 
        dropouts -> list with prob of dropout between layers
        """
        self.layers = nn.ModuleList()
        self.name = "fc"

        for i in range(nr_fc):

            out_features = fc_size[i] #fc_size must always end w/ 62 (nr of classes)

            #fc layer
            fc_layer = nn.Linear(in_features=in_features, out_features=out_features)
            in_features = fc_size[i]
            self.layers.append(fc_layer)

            self.name += f"_{in_features}"
            if i < nr_fc -1 : #not in final layer

                #activation function
                act_function = getattr(nn, act_funs[i])()  
                self.layers.append(act_function)

                #dropout prob
                if dropouts[i] > 0:
                    self.layers.append(nn.Dropout(dropouts[i]))

                self.name += f"{act_funs[i]}{dropouts[i]}"




def models_iterator(nr_conv, filters, maxpool_batchnorm, nr_fconnected, fc_sizes, act_funs, drops, lr):
    models_to_train = []
    configurations = list(itertools.product(maxpool_batchnorm, drops))
    for nr_c in range(nr_conv):
        for nr_fc in range(1, nr_fconnected):
            for config in configurations:

                maxpool_batchnorm, drops = config

                if not(nr_fc == 1 and drops == [0]*3): #avoid making 2 copied models in the case where there is only 1 layer: both "drops" options will be the same [] !! 
                    nr_filters = filters[:nr_c]

                    conv_layers = Conv(nr_conv=nr_c, nr_filters=nr_filters, maxpool_batchnorm=maxpool_batchnorm)

                    fc_size=fc_sizes[4-nr_fc:] #want to slice from the end
                    act_fun=act_funs[:nr_fc-1]
                    dropouts = drops[4-nr_fc:] #want to slice from the end
                    
                    fc_layers = FC(nr_fc=nr_fc, fc_size=fc_size, act_funs=act_fun, dropouts=dropouts, in_features=conv_layers.finaldim)
                    
                    # Create the model using the CNN constructor
                    model = CNN(conv_layers=conv_layers, fc_layers=fc_layers, num_classes=62)  # Assuming 62 classes as an example
                    
                    # Store the model and its parameters in the list
                    model_info = {
                        "name": f"{model.name}", 
                        "model": model,
                        "params": {"lr": lr}
                    }
                    
                    models_to_train.append(model_info)

    return models_to_train


filters = [8, 16, 32, 64]
maxpool_batchnorm = ["False", "True"]
fc_sizes = [320, 160, 80, 62]
lr = 0.01
act_funs = ['ReLU'] * 3
drops =[ [0.5, 0.2, 0.2], [0]*3]


models_to_train = models_iterator(4, filters, maxpool_batchnorm, 5, fc_sizes, act_funs, drops, lr)

decent_conv = Conv(3, [16, 32, 64])
decent_fc = FC(3, [300, 150, 62], ['ReLU']* 2, dropouts= [0.5, 0.2])
decent_model = CNN(decent_conv, decent_fc, 62)


print(f"list of {len(models_to_train)} models generated!!")
print(models_to_train)

# possible optimizers: {SGD, ADAM, RMSprop}
#atenção a early stopping: nas primeiras epochs não faz sentido
#some pretrained models: {MobileNetV2, ResNet50, EfficientNet}
