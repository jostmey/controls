#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import torch, torchmetrics, torchvision
from epoch_summary_writer import EpochSummaryWriter

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--batch', help='Batch size', type=int, default=1024)
parser.add_argument('--step', help='Step size', type=float, default=0.001)
parser.add_argument('--epochs', help='Number of passes over the dataset', type=int, default=1024)
parser.add_argument('--device', help='Examples are cuda:0 or cpu', type=str, default='cpu')
parser.add_argument('--basepath', help='Base path for saving checkpoint and logs', type=str, default=None)
args = parser.parse_args()

##########################################################################################
# Settings
##########################################################################################

device = torch.device(args.device)

seed = 90843
generator = torch.Generator()
generator.manual_seed(seed)

##########################################################################################
# Load data
##########################################################################################

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torch.nn.Flatten(start_dim=0, end_dim=2)
    ]
)

dataset = torchvision.datasets.MNIST(root='datasets', train=True, download=True, transform=transform)
dataset_test = torchvision.datasets.MNIST(root='datasets', train=False, download=True, transform=transform)

num = len(dataset)
num_train = int(5/6*num)
num_val = num-num_train

dataset_train, dataset_val = torch.utils.data.random_split(dataset, [ num_train, num_val ], generator=generator)

sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True, generator=generator)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
sampler_test = torch.utils.data.SequentialSampler(dataset_test)

loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch, sampler=sampler_train)
loader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.batch, sampler=sampler_val)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch, sampler=sampler_test)

##########################################################################################
# Model
##########################################################################################

class Model(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, p=0.5, **kwargs):
        super().__init__(**kwargs)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_inputs),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p),
            torch.nn.BatchNorm1d(num_inputs)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_inputs),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p),
            torch.nn.BatchNorm1d(num_inputs)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_inputs),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p),
            torch.nn.BatchNorm1d(num_inputs)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_outputs),
            torch.nn.BatchNorm1d(num_outputs)
        )

    def forward(self, x):
        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return z4

##########################################################################################
# Instantiate model, optimizer, and metrics
##########################################################################################

logger = EpochSummaryWriter(log_dir=args.basepath)

model = Model(28**2, 10).to(device)
softmax = torch.nn.Softmax(dim=1)

optimizer = torch.optim.Adam(model.parameters(), lr=args.step)

loss = torch.nn.CrossEntropyLoss()
accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

##########################################################################################
# Run the model
##########################################################################################

i_best = -1
e_best = 1.0e8
a_best = 0.0
state_best = {}

logger.add_custom_scalars(
    {
        'merged': {
            'error'   : [ 'Multiline', [ 'error/train',    'error/val',    'error/test' ]    ],
            'accuracy': [ 'Multiline', [ 'accuracy/train', 'accuracy/val', 'accuracy/test' ] ],
            'T'       : [ 'Multiline', [ 'T/train',        'T/val',        'T/test' ]        ],
        }
    }
)

for i in range(args.epochs):

    model.train()
    for xs_batch, ys_batch in iter(loader_train):
        xs_batch = xs_batch.to(device)
        ys_batch = ys_batch.to(device)
        ls_batch = model(xs_batch)
        ps_batch = softmax(ls_batch)
        e_batch = loss(ls_batch, ys_batch)
        a_batch = accuracy(ps_batch, ys_batch)
        optimizer.zero_grad()
        e_batch.backward()
        optimizer.step()
        logger.update_scalar('error', e_batch/0.693147)
        logger.update_scalar('accuracy', 100.0*a_batch)
    logger.add_update_scalars(i, tag_suffix='train')

    e_val = 0.0
    model.eval()
    with torch.no_grad():
        for xs_batch, ys_batch in iter(loader_val):
            xs_batch = xs_batch.to(device)
            ys_batch = ys_batch.to(device)
            ls_batch = model(xs_batch)
            ps_batch = softmax(ls_batch)
            e_batch = loss(ls_batch, ys_batch)
            a_batch = accuracy(ps_batch, ys_batch)
            fraction = float(ys_batch.shape[0])/float(len(dataset_val))
            e_val += fraction*e_batch.detach()
            logger.update_scalar('error', e_batch/0.693147)
            logger.update_scalar('accuracy', 100.0*a_batch)
        if e_val < e_best:
            i_best = i
            e_best = e_val
            state_best = model.state_dict()
    logger.add_update_scalars(i, tag_suffix='val')

    logger.flush()

model.load_state_dict(state_best)
model.eval()
with torch.no_grad():
    for xs_batch, ys_batch in iter(loader_test):
        xs_batch = xs_batch.to(device)
        ys_batch = ys_batch.to(device)
        ls_batch = model(xs_batch)
        ps_batch = softmax(ls_batch)
        e_batch = loss(ls_batch, ys_batch)
        a_batch = accuracy(ps_batch, ys_batch)
        logger.update_scalar('error', e_batch/0.693147)
        logger.update_scalar('accuracy', 100.0*a_batch)
    logger.add_update_scalars(i_best, tag_suffix='test')

##########################################################################################
# Save the model
##########################################################################################

if args.basepath is not None:
    torch.save(model.state_dict(), args.basepath+'.p')
