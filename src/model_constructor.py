import torch.nn as nn
import itertools


class CNN(nn.Module):
    """
    Constructor receives: layer objects and num_classes. can be conv layer or fc layer.
    Each is defined below and has multiple options for architecture.
    """

    def __init__(self, conv_layers, fc_layers, num_classes, lr, optim):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(*conv_layers.layers)
        self.fc_layers = nn.Sequential(*fc_layers.layers)
        self.num_classes = num_classes
        self.lr = lr
        safe_lr = str(lr).replace(".", "")
        self.optim = optim
        self.name = f"{conv_layers.name}{fc_layers.name}_lr{safe_lr}_{optim}"

    def forward(self, input):  # input will be of form (Batch size, 3, 32, 32)
        # conv layers

        input = self.conv_layers(input)

        # flatten to fit fully connected layer
        input = input.reshape(((input.shape[0], input.shape[1] * input.shape[2] * input.shape[3])))

        # fully connected layers
        input = self.fc_layers(input)

        return input


class Conv:
    def __init__(self, nr_conv=0, nr_filters=[], maxpool_batchnorm=True):
        """
        nr_conv -> nr of convolution layers
        nr_filters -> list with filters for each layer
        maxpool_batchnorm -> if true, both batch normalization and maxpooling are exacuted after each convolution
        """
        self.name = f"conv{nr_conv}{maxpool_batchnorm}"
        self.layers = nn.ModuleList()

        input_size = 32
        channels = 3  # rgb

        for i in range(nr_conv):
            # convolution does not alter dimensions
            conv_layer = nn.Conv2d(
                in_channels=channels,
                out_channels=nr_filters[i],
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            )

            channels = nr_filters[i]
            self.layers.append(conv_layer)

            self.name += f"_{channels}"

            if maxpool_batchnorm:
                batchnorm_layer = nn.BatchNorm2d(channels)
                self.layers.append(batchnorm_layer)

            self.layers.append(nn.ReLU())

            if maxpool_batchnorm:
                maxpool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
                self.layers.append(maxpool_layer)
                input_size /= 2  # maxpooling halves dimensions

        self.finaldim = int(input_size * input_size * channels)


class FC:
    def __init__(
        self, nr_fc=0, fc_size=[], act_funs=[], dropouts=[], in_features=0, batchnorm=True, num_classes=10
    ):
        """
        nr_fc -> nr of fully connected layers. if =0, then it is just linear
        fc_size -> list with nr of nodes for each layer
        act_funs -> list with activation function name (needs to correspond to pytorch name) for each layer
        dropouts -> list with prob of dropout between layers
        """
        self.layers = nn.ModuleList()
        self.name = f"fc{batchnorm}"

        for i in range(nr_fc):
            out_features = fc_size[i]

            # fc layer
            fc_layer = nn.Linear(in_features=in_features, out_features=out_features)
            in_features = fc_size[i]
            self.layers.append(fc_layer)

            self.name += f"_{in_features}"

            if batchnorm:
                # Add Batch Normalization
                batchnorm_layer = nn.BatchNorm1d(out_features)
                self.layers.append(batchnorm_layer)

            # activation function
            act_function = getattr(nn, act_funs[i])()
            self.layers.append(act_function)

            # dropout prob
            if dropouts[i] > 0:
                self.layers.append(nn.Dropout(dropouts[i]))
            safe_dropout = str(dropouts[i]).replace(".", "")
            self.name += f"{act_funs[i]}{safe_dropout}"

        # last layer
        self.layers.append(nn.Linear(in_features=in_features, out_features=num_classes))


class VGG(nn.Module):
    """
    Constructor receives: layer objects and num_classes. can be conv layer or fc layer.
    Each is defined below and has multiple options for architecture.
    """

    def __init__(self, num_classes, drop):
        super(VGG, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout(drop),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding="same"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding="same"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout(drop),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding="same"),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding="same"),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout(drop),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        )

        self.fc_layers = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_classes))
        self.num_classes = num_classes

    def forward(self, input):  # input will be of form (Batch size, 3, 32, 32)
        input = self.conv_layers(input)
        input = input.reshape(((input.shape[0], input.shape[1] * input.shape[2] * input.shape[3])))
        input = self.fc_layers(input)

        return input


def vgg_iterator(lrs, optimizers):
    models_vgg = []
    for lr in lrs:
        for optimizer in optimizers:
            model_name = f"vgg_cifar_{lr}_{optimizer}"
            model_info = {
                "name": model_name,
                "model": VGG(10, 0.5),
                "params": {"lr": lr, "optimizer": optimizer},
            }
            models_vgg.append(model_info)

    return models_vgg


def models_iterator(depths, filters_sizes, optimizers, drops, lrs):
    models_to_train = []
    for depth in depths:
        fs = filters_sizes[str(depth)]
        d = drops[str(depth)]
        configurations = list(itertools.product(fs, optimizers, d, lrs))

        for config in configurations:
            filters_size, optimizer, drop, lr = config

            nr_filters = filters_size[:depth]
            conv_layers = Conv(nr_conv=depth, nr_filters=nr_filters, maxpool_batchnorm=True)
            fc_size = filters_size[depth:]
            act_fun = ["ReLU"] * depth
            dropouts = drop
            fc_layers = FC(
                nr_fc=depth,
                fc_size=fc_size,
                act_funs=act_fun,
                dropouts=dropouts,
                in_features=conv_layers.finaldim,
                num_classes=10,
                batchnorm=True,
            )

            # Create the model using the CNN constructor
            model = CNN(
                conv_layers=conv_layers, fc_layers=fc_layers, num_classes=10, lr=lr, optim=optimizer
            )

            # Store the model and its parameters in the list
            model_info = {
                "name": f"{model.name}",
                "model": model,
                "params": {"lr": lr, "optimizer": optimizer},
            }

            models_to_train.append(model_info)

    models_vgg = vgg_iterator(lrs, optimizers)
    models_to_train += models_vgg
    print(f"list of {len(models_to_train)} models generated!!")

    return models_to_train
