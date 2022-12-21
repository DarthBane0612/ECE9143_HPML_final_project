import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable


import os
import argparse
import time


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')


parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')

parser.add_argument('--batchsize', type=int, default='32',
                    help='choose batch size')

parser.add_argument('--numepoch', type=int, default='2',
                    help='choose number of epochs')

parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer choice')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda training')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


start_epoch = 1

datapre_time_start = time.perf_counter()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers)

datapre_time_end = time.perf_counter()
datapreprocess_time = datapre_time_end - datapre_time_start
print("Data preprocessing time: ", datapreprocess_time)

net = VGG('VGG19')



class Trainer(object):
    def __init__(self,
                 net,
                 fp16=True,
                 loss_scaling=True):
        self.net = net
        self.loss_scaling = loss_scaling
        self.fp16_mode = fp16

        self.best_acc = 0
        self.best_epoch = 0
        self._LOSS_SCALE = 128.0

        self.net = self.net.cuda()
        if self.fp16_mode:
            self.net = self.network_to_half(self.net)
            self.model_params, self.master_params = self.prep_param_list(
                self.net)

        
        self.net = nn.DataParallel(self.net)

    def prep_param_list(self, model):
        model_params = [p for p in model.parameters() if p.requires_grad]
        master_params = [p.detach().clone().float() for p in model_params]

        for p in master_params:
            p.requires_grad = True

        return model_params, master_params
    
    def master_params_to_model_params(self, model_params, master_params):
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)


    def model_grads_to_master_grads(self, model_params, master_params):
        for model, master in zip(model_params, master_params):
            if master.grad is None:
                master.grad = Variable(master.data.new(*master.data.size()))
            master.grad.data.copy_(model.grad.data)


    def BN_convert_float(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.float()
        for child in module.children():
            self.BN_convert_float(child)
        return module


    class tofp16(nn.Module):
        def __init__(self):
            super(Trainer.tofp16, self).__init__()

        def forward(self, input):
            return input.half()

    def network_to_half(self, network):
        return nn.Sequential(self.tofp16(), self.BN_convert_float(network.half()))

    total_time = 0

    def train(self, epoch, trainloader, lr):
        self.net.train()

        train_loss, correct, total = 0, 0, 0
        total_time_start = time.perf_counter()
        first_time = 0
        load_time_start = time.perf_counter()
        total_train_time = 0

        
        if not hasattr(self, 'optimizer'):
            if self.fp16_mode:
                self.optimizer = optim.SGD(self.master_params, lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        scheduler.step()
        criterion = nn.CrossEntropyLoss()
        best_epoch_accuracy = 0
        for idx, (inputs, targets) in enumerate(trainloader):
            if first_time == 0:
                first_time = 1
                load_time_end = time.perf_counter()
                print("data loading time: ", load_time_end - load_time_start)

            train_time_start = time.perf_counter()
            inputs, targets = inputs.cuda(), targets.cuda()
            self.net.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)
            if self.loss_scaling:
                loss = loss * self._LOSS_SCALE
            loss.backward()

            if self.fp16_mode:
                self.model_grads_to_master_grads(self.model_params,
                                                 self.master_params)
                if self.loss_scaling:
                    for params in self.master_params:
                        params.grad.data = params.grad.data / self._LOSS_SCALE
                self.optimizer.step()
                self.master_params_to_model_params(self.model_params, self.master_params)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (targets == predicted).sum().item()

            train_time_end = time.perf_counter()
            total_train_time += train_time_end - train_time_start

            accuracy_this = 100.*correct/total
            if accuracy_this > best_epoch_accuracy:
                best_epoch_accuracy = accuracy_this
        total_time_end = time.perf_counter()
        
        print("training accuracy for current epoch: ", best_epoch_accuracy, "%")
        print("Training time for current epoch(w/o data loading): ", total_train_time)
        print("total running time for current epoch: ", total_time_end - total_time_start)

    def training(self, traindataloader, epoch, lr):
        self.best_acc = 0.0
        for i in range(epoch):
            print('\nEpoch: %d' % (i + 1))
            self.train(i, traindataloader, lr)


trainer = Trainer(net)
trainer.training(trainloader, args.numepoch, 0.1)

