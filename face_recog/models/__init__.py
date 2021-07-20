
from .face_model import *
# from .dynamic_resnet import *

def get_model(model_name:str):
    ''' return a model by the name '''
    net = None
    
    if model_name == 'ResNet18':
        net = ResNet18()
    elif model_name == 'ResNet34':
        net = ResNet34()
    elif model_name == 'ResNet50':
        net = ResNet50()
    elif model_name == 'ResNet101':
        net = ResNet101()
    elif model_name == 'ResNet152':
        net = ResNet152()
    elif model_name == 'VGG':
        net = VGG('VGG19')
    elif model_name == 'ResNeXt29_8x64d':
        net = ResNeXt29_8x64d()
    elif model_name == 'WideResNet20-8':
        net = wideresnet(depth=20, widen_factor=8)
    elif model_name == 'WideResNet44-8':
        net = wideresnet(depth=44, widen_factor=8)
    elif model_name == 'WideResNet28-12':
        net = wideresnet(depth=28, widen_factor=12)
    elif model_name == 'DenseNet121':
        net = DenseNet121()
    elif model_name == 'DenseNet169':
        net = DenseNet169()
    elif model_name == 'DenseNet201':
        net = DenseNet201()
    elif model_name == 'DenseNet161':
        net = DenseNet161()
    elif model_name == 'Dynamic_Resnet18':
        net = resnet18()
    elif model_name == 'XunFeiNet':
        net = XunFeiNet()
    assert net is not None, 'Please make sure the name you take is right!'
    return net