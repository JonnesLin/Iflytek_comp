import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import argparse
import wandb
from models import *
from data_process import *
from utils import progress_bar, set_seed, cal_ent_and_draw
import torch.nn.functional as F
import warnings
import yaml
from double_dataset import DoubleDataset
import sys
import imgaug
from typing import Optional, Sequence
from torch import Tensor
from data_process import Face_dataset
from warmup_scheduler import GradualWarmupScheduler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best test accuracy
best_acc = 0
# start from epoch 0 or last checkpoint epoch
start_epoch = 0

##################
# Get Parameters #
##################
with open('conf/basic_conf.yml') as f:
    args = yaml.load(f)


############
# set seed #
############
set_seed(args['seed'])


################
# Augmentation #
################
transform_train = get_train_transform(args)
transform_test = get_test_transform(args)

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)

##########################
# Dataset and DataLoader #
##########################
trainset = Face_dataset(True, transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args['batch_size'], shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

testset = Face_dataset(False, transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

#########
# Wandb #
#########
Net_Name = args['project_name']
checkpoint = args['project_name']
wandb.init(project=args['project_name'],
           # group=wandb.util.generate_id(),
           group=str('batch:' + str(args['batch_size']) + str('_change_auxi')),
           config={
               # optimizer
               "lr": args['lr'],
               "batch_size": args['batch_size']
           })

#########
# Model #
#########
net = get_model('XunFeiNet')
net = net.to(device)
wandb.watch(net)

# AMP if it's necessary(impart performance)
if args['amp']:
    scaler = GradScaler()


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

##################################################
# Definition of the Focal loss and the Soft Loss #
##################################################
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """
					

    def __init__(self,
                 alpha: Optional[Tensor] = torch.tensor([1.063, 4.468, 1.021, 0.441, 0.787, 0.815, 1.406]).cuda(),
                 gamma: float = 0.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def soft_loss(outputs, targets, norm=None):
    log_softmax_outputs = F.log_softmax(outputs / args['temperature'], dim=1)
    softmax_targets = F.softmax(targets / args['temperature'], dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

#####################################
# Criterion, Optimizer and Scheduler#
#####################################
ce = FocalLoss().cuda()
optimizer = optim.AdamW(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epoch'], eta_min=args['eta_min'])
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    ensemble_pred_correct = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Get X and Y
        inputs, targets = inputs.to(device), targets.to(device)
        # To soft  label!
        one_hot_label = (1-args['alpha'])*F.one_hot(targets, num_classes=args['num_classes']) + args['alpha']/args['num_classes']
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        ensemble = outputs
        # The loss includes focal loss and softed loss
        loss = ce(outputs, targets) + soft_loss(outputs, one_hot_label)
        
        
        if args['amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if args['amp']:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, ensemble_pred = ensemble.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        ensemble_pred_correct += ensemble_pred.eq(targets).sum().item()

        progress_bar(batch_idx,
                     len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | Ensemble: %.3f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*ensemble_pred_correct/total)
                    )
    wandb.log({'train_acc': 100. * correct / total, 'train_loss': train_loss / (batch_idx + 1),
               'train_ensemble_pred': 100.*ensemble_pred_correct/total})


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ensemble_pred_correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = net(inputs)
            ensemble = outputs
            
            
            loss = ce(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, ensemble_pred = ensemble.max(1)
            
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
            ensemble_pred_correct += ensemble_pred.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Ensemble: %.3f'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                            100. * ensemble_pred_correct / total))

    wandb.log({'test_acc': 100. * correct / total, 'test_loss': test_loss / (batch_idx + 1),
               'test_ensemble_pred': 100.*ensemble_pred_correct/total})

    

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint):
            os.mkdir(checkpoint)
        torch.save(state, './' + checkpoint + '/' + str(args['project_name']) + 'ckpt.pth')
        best_acc = acc


def predict(test_loader, model, tta=10):
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
    
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta


for epoch in range(start_epoch, start_epoch + args['epoch']):
    train(epoch)
    test(epoch)
    scheduler.step()


test_jpg = glob.glob('data/Datawhale_人脸情绪识别_数据集/test/*')
test_jpg = np.array(test_jpg)

test_loader = torch.utils.data.DataLoader(
    Face_pred_dataset(
        test_jpg,
        transforms.Compose([
            transforms.Resize((args['image_size'], args['image_size'])),
            # transforms.RandomAffine(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(args['image_size'], padding=16),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ), batch_size=64, shuffle=False, num_workers=8, pin_memory=True
)


    
test_pred = predict(test_loader, net)


cls_name = np.array(['angry', 'disgusted', 'fearful', 'happy','neutral', 'sad', 'surprised'])
submit_df = pd.DataFrame({'name': test_jpg, 'label': cls_name[test_pred.argmax(1)]})
submit_df['name'] = submit_df['name'].apply(lambda x: x.split('/')[-1])

submit_df = submit_df.sort_values(by='name')
submit_df.to_csv('pytorch_submit.csv', index=None)



print('--------------------   Best ACC Of Teacher Model   --------------------')
print(best_acc)
