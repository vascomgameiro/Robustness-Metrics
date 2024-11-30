from torch import nn


def modify_last_layer(model, model_name, num_classes):
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("mobilenet_v3"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported for modification.")
    return model


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):  # Adjust for 62 classes
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 3 channels (RGB)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Adjust the fully connected layer based on the input size
        # Assuming input size is 64x64 (or similar), you need to calculate the flattened size after pooling
        # Example: if input images are 64x64, the output size after pooling layers would be 256 * 8 * 8
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutions and activations
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)  # Flattening the feature maps
        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer

        return x
