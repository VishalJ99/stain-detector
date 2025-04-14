import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes, pretrained=True):
    """
    Get a model from torchvision models with custom classification head
    
    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet50')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Model with custom classification head
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Modify the final classification layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=pretrained)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    return model


class StainClassifier(nn.Module):
    """Wrapper class for the stain classification model"""
    
    def __init__(self, config):
        """
        Initialize the model based on configuration
        
        Args:
            config: Configuration object with model parameters
        """
        super(StainClassifier, self).__init__()
        
        self.model_name = config.model.name
        self.num_classes = config.data.num_classes
        self.pretrained = config.model.pretrained
        
        self.model = get_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)