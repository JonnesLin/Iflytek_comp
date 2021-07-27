from .autoaugment import CIFAR10Policy, Cutout, ImageNetPolicy
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_train_transform(args):
    ''' return train data transform 
    
        parameteres:
            args: ArgumentParser
        return:
            data transforme: torchvision.transforms
    '''
    transform_ways = []
    # add augmentation by parameteres
    transform_ways.append(transforms.Resize((args['image_size'], args['image_size'])))
    if args['train_random_crop']:
        transform_ways.append(
            transforms.RandomCrop(args['image_size'], padding=args['train_random_crop_padding']))
    if args['train_random_horizontalFlip']:
        transform_ways.append(
            transforms.RandomHorizontalFlip(args['train_random_horizontalFlip_prob']))
    if args['train_cifar10_policy']:
        transform_ways.append(CIFAR10Policy())
    if args['imagenet']:
        transform_ways.append(ImageNetPolicy())
    
    # if args['train_cutout']:
    #     transform_ways.append(
    #         Cutout(n_holes=args['train_cutout_n'], length=args['train_cutout_length']))
    
    # transform_ways.append(transforms.Normalize(eval(args['train_normlization'])[0], eval(args['train_normlization'])[1]))
    
    data_transform = transforms.Compose([
        *transform_ways,
        transforms.ToTensor(),
        Cutout(n_holes=args['train_cutout_n'], length=args['train_cutout_length']),
        transforms.Normalize(eval(args['train_normlization'])[0], eval(args['train_normlization'])[1])
    ])
    
    return data_transform


def get_test_transform(args):
    ''' return test data transform 
    
        parameteres:
            args: ArgumentParser
        return:
            data transforme: torchvision.transforms
    '''
    transform_ways = []
    transform_ways.append(transforms.Resize((args['image_size'], args['image_size'])))
    transform_ways.append(transforms.ToTensor())
    transform_ways.append(transforms.Normalize(eval(args['test_normlization'])[0], eval(args['test_normlization'])[1]))
    
    data_transform = transforms.Compose([*transform_ways])
    
    return data_transform
