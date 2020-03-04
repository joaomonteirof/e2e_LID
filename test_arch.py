from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_
from utils.losses import AMSoftmax, Softmax

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['mfcc', 'fb', 'resnet_fb', 'resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'lcnn9_mfcc', 'lcnn29_mfcc', 'TDNN', 'FTDNN', 'all'], default='fb', help='Model arch according to input type')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=13, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--softmax', action='store_true', default=False, help='Test also sm layers')
args = parser.parse_args()

if args.model == 'mfcc' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.cnn_lstm_mfcc(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('mfcc', mu.size())
if args.model == 'fb' or args.model == 'all':
	batch = torch.rand(3, 1, 40, 400)
	model = model_.cnn_lstm_fb(n_z=args.latent_size)
	mu = model.forward(batch)
	print('fb', mu.size())
if args.model == 'resnet_fb' or args.model == 'all':
	batch = torch.rand(3, 1, 40, 400)
	model = model_.ResNet_fb(n_z=args.latent_size)
	mu = model.forward(batch)
	print('resnet_fb', mu.size())
if args.model == 'resnet_mfcc' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.ResNet_mfcc(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_mfcc', mu.size())
if args.model == 'resnet_lstm' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.ResNet_lstm(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_lstm', mu.size())
if args.model == 'resnet_stats' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.ResNet_stats(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('resnet_stats', mu.size())
if args.model == 'lcnn9_mfcc' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.lcnn_9layers(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('lcnn9_mfcc', mu.size())
if args.model == 'lcnn29_mfcc' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.lcnn_29layers_v2(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('lcnn29_mfcc', mu.size())
if args.model == 'TDNN' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.TDNN(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('TDNN', mu.size())
if args.model == 'FTDNN' or args.model == 'all':
	batch = torch.rand(3, 1, args.ncoef, 400)
	model = model_.FTDNN(n_z=args.latent_size, ncoef=args.ncoef)
	mu = model.forward(batch)
	print('FTDNN', mu.size())

if args.softmax:
	batch = torch.rand(3, mu.size(0))
	batch_labels = torch.randint(low=0, high=10, size=(mu.size(0),))

	amsm = AMSoftmax(input_features=batch.size(1), output_features=10)
	sm = Softmax(input_features=batch.size(1), output_features=10)

	print('amsm', amsm(batch, batch_labels).size())
	print('sm', sm(batch).size())
