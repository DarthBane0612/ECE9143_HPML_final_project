import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed
import torchvision
import torchvision.transforms as transforms

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


#distributed learning
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

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

parser.add_argument('--dp', action='store_true', default=False,
                    help='utilize data parallel')

parser.add_argument('--dl', action='store_true', default=False,
                    help='utilize distributed learning')

parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--ngpu', type=int, default=2,
                    help='specify number of gpus')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

best_acc = 0  # best test accuracy
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

    
if args.dl and device == 'cuda':
    world_size = args.ngpu
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=args.local_rank,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=args.ngpu,
        rank=args.local_rank,
    )
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    torch.cuda.set_device(args.local_rank)
    net = net.to(device)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )


elif args.dp:
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

else:
    net = net.to(device)





criterion = nn.CrossEntropyLoss()

if args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
elif args.optimizer == "sgdnes":
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4, nesterov = True)
elif args.optimizer == "adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=0.1,
                      weight_decay=5e-4)
elif args.optimizer == "adadelta":
    optimizer = optim.Adadelta(net.parameters(), lr=0.1,
                      weight_decay=5e-4)
elif args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=0.1,
                      weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


total_time = 0
best_accuracy = 0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global total_time, total_loss, best_accuracy

    #start total running time for a epoch
    total_time_start = time.perf_counter()

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    #total training loss for 1 epoch
    train_loss_epoch = 0

    #for loading time calculation
    first_time = 0
    load_time_start = time.perf_counter()

    #for total train time for a epoch
    total_train_time = 0;

    #top 1 accuracy for current epoch
    best_epoch_accuracy = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #get data loading time
        if first_time == 0:
            first_time = 1
            load_time_end = time.perf_counter()
            print("data loading time: ", load_time_end - load_time_start)

        #start train time count for 1 batch
        train_time_start = time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #end train time count for 1 batch
        train_time_end = time.perf_counter()

        #add to the total train time
        total_train_time += train_time_end - train_time_start

        #get total train loss for current epoch
        train_loss_epoch += train_loss

        

        #check whether this batch accuracy is better than current epoch best accuracy
        accuracy_this = 100.*correct/total
        if accuracy_this > best_epoch_accuracy:
            best_epoch_accuracy = accuracy_this

    #end count for total running time for a epoch
    total_time_end = time.perf_counter()
    

    #get average training time for current epoch
    average_epoch_training_time = total_train_time / args.batchsize

    #get average training loss for current epoch
    average_epoch_loss = train_loss_epoch / args.batchsize

    #get total training time epoch
    total_time += total_train_time

    #get top 1 accuracy
    if best_epoch_accuracy > best_accuracy:
        best_accuracy = best_epoch_accuracy


    print("Training time for current epoch(w/o data loading): ", total_train_time)
    print("Total Inference time: ", total_time_end - total_time_start - total_train_time - (load_time_end - load_time_start))
    print("total running time for current epoch: ", total_time_end - total_time_start)
    print("Average training loss for current epoch: ", average_epoch_loss)
    print("Average training time for current epoch(w/o data loading): ", average_epoch_training_time)
    print("Training accuracy: ", best_epoch_accuracy, "%")




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


for epoch in range(start_epoch, start_epoch+args.numepoch):
    train(epoch)
    test(epoch)
    scheduler.step()
