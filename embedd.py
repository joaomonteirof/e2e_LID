import argparse
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp, open_or_fd, write_vec_flt
import model as model_
import scipy.io as sio

def set_device(trials=10):
	a = torch.rand(1)

	for i in range(torch.cuda.device_count()):
		for j in range(trials):

			torch.cuda.set_device(i)
			try:
				a = a.cuda()
				print('GPU {} selected.'.format(i))
				return
			except:
				pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def prep_feats(data_):

	#data_ = ( data_ - data_.mean(0) ) / data_.std(0)

	features = data_.T

	if features.shape[1]<50:
		mul = int(np.ceil(50/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :50]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute embeddings')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to input data')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--model', choices=['mfcc', 'fb', 'resnet_fb', 'resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'lcnn9_mfcc', 'lcnn29_mfcc', 'TDNN', 'FTDNN'], default='fb', help='Model arch according to input type')
	parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
	parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	print('Cuda Mode is: {}'.format(args.cuda))

	if args.cuda:
		set_device()

	if args.model == 'mfcc':
		model = model_.cnn_lstm_mfcc(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
	elif args.model == 'fb':
		model = model_.cnn_lstm_fb(n_z=args.latent_size, proj_size=None)
	elif args.model == 'resnet_fb':
		model = model_.ResNet_fb(n_z=args.latent_size, proj_size=None)
	elif args.model == 'resnet_mfcc':
		model = model_.ResNet_mfcc(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
	elif args.model == 'resnet_lstm':
		model = model_.ResNet_lstm(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
	elif args.model == 'resnet_stats':
		model = model_.ResNet_stats(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
	elif args.model == 'lcnn9_mfcc':
		model = model_.lcnn_9layers(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
	elif args.model == 'lcnn29_mfcc':
		model = model_.lcnn_29layers_v2(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
	elif args.model == 'TDNN':
		model = model_.TDNN(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)
	elif args.model == 'FTDNN':
		model = model_.FTDNN(n_z=args.latent_size, proj_size=None, ncoef=args.ncoef)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=False)

	model.eval()

	if args.cuda:
		model = model.cuda()

	scp_list = glob.glob(args.path_to_data + '*.scp')

	if len(scp_list)<1:
		print('Nothing found at {}.'.format(args.path_to_data))
		exit(1)

	print('Start of data embeddings computation')

	embeddings = {}

	for file_ in scp_list:

		data = { k:m for k,m in read_mat_scp(file_) }

		for i, utt in enumerate(data):

			print('Computing embedding for utterance '+ utt)

			feats = prep_feats(data[utt])

			try:
				if args.cuda:
					feats = feats.cuda()
					model = model.cuda()

				emb = model.forward(feats)

			except:
				feats = feats.cpu()
				model = model.cpu()

				emb = model.forward(feats)

			embeddings[utt] = emb.detach().cpu().numpy().squeeze()

	print('Storing embeddings in output file')

	out_name = args.path_to_data.split('/')[-2]
	file_name = args.out_path+out_name+'.ark'

	if os.path.isfile(file_name):
		os.remove(file_name)
		print(file_name + ' Removed')

	with open_or_fd(file_name,'wb') as f:
		for k,v in embeddings.items(): write_vec_flt(f, v, k)

	print('End of embeddings computation.')
