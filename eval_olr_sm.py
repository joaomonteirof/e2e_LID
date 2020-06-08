import argparse
import numpy as np
import torch
from kaldi_io import read_mat_scp
from sklearn import metrics
import scipy.io as sio
import model as model_
import glob
import pickle
import torch.nn.functional as F

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

def prep_feats(data_, min_nb_frames=50):

	features = data_.T

	if features.shape[1]<min_nb_frames:
		mul = int(np.ceil(min_nb_frames/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :min_nb_frames]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

def compute_metrics(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	t = np.nanargmin(np.abs(fnr-fpr))

	eer_threshold = thresholds[t]

	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	auc = metrics.auc(fpr, tpr)

	avg_precision = metrics.average_precision_score(y, y_score)

	pred = np.asarray([1 if score > eer_threshold else 0 for score in y_score])
	acc = metrics.accuracy_score(y ,pred)

	return eer, auc, avg_precision, acc, eer_threshold

def read_trials(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	enroll_spk_list, test_utt_list, labels_list = [], [], []

	for line in utt_labels:
		enroll_spk, test_utt, label = line.split(' ')
		enroll_spk_list.append(enroll_spk)
		test_utt_list.append(test_utt)
		labels_list.append(1 if label=='target\n' else 0)

	return enroll_spk_list, test_utt_list, labels_list

def read_spk2utt(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to input data')
	parser.add_argument('--sil-data', type=str, default='./data/', metavar='Path', help='Path to input data with silence')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--model', choices=['mfcc', 'fb', 'resnet_fb', 'resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'lcnn9_mfcc', 'lcnn29_mfcc', 'TDNN', 'TDNN_multipool', 'FTDNN'], default='fb', help='Model arch according to input type')
	parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
	parser.add_argument('--ncoef', type=int, default=13, metavar='N', help='number of MFCCs (default: 13)')
	parser.add_argument('--scores-file', type=str, default='./scores.out', metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	labels_dict = {'Kazak':0, 'Tibet':1, 'Uyghu':2, 'ct':3, 'id':4, 'ja':5, 'ko':6, 'ru':7, 'vi':8, 'zh':9}

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	print('Cuda Mode is: {}\n'.format(args.cuda))

	if args.cuda:
		set_device()

	if args.model == 'mfcc':
		model = model_.cnn_lstm_mfcc(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'fb':
		model = model_.cnn_lstm_fb(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())))
	elif args.model == 'resnet_fb':
		model = model_.ResNet_fb(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())))
	elif args.model == 'resnet_mfcc':
		model = model_.ResNet_mfcc(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'resnet_lstm':
		model = model_.ResNet_lstm(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'resnet_stats':
		model = model_.ResNet_stats(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'lcnn9_mfcc':
		model = model_.lcnn_9layers(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'lcnn29_mfcc':
		model = model_.lcnn_29layers_v2(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'TDNN':
		model = model_.TDNN(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'TDNN_multipool':
		model = model_.TDNN_multipool(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)
	elif args.model == 'FTDNN':
		model = model_.FTDNN(n_z=args.latent_size, proj_size=len(list(labels_dict.keys())), ncoef=args.ncoef)

	print(model)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=True)

	model.eval()

	if args.cuda:
		model = model.cuda()

	files_list = glob.glob(args.data_path+'*.scp')

	data = None

	for file_ in files_list:
		if data is None:
			data = { k:v for k,v in read_mat_scp(file_) }
		else:
			for k,v in read_mat_scp(file_):
				data[k] = v

	files_list = glob.glob(args.sil_data+'*.scp')

	sil_data = None

	for file_ in files_list:
		if sil_data is None:
			sil_data = { k:v for k,v in read_mat_scp(file_) }
		else:
			for k,v in read_mat_scp(file_):
				sil_data[k] = v

	speakers_enroll, utterances_test, labels = read_trials(args.trials_path)

	print('\nAll data ready. Start of scoring')

	scores = []
	out_data = []

	for i in range(len(labels)):

		test_utt = utterances_test[i]

		try:

			test_utt_data = prep_feats(data[test_utt])

			if args.cuda:
				test_utt_data = test_utt_data.cuda()
				model = model.cuda()
			out_sm = model.out_proj( model.forward(test_utt_data) ).detach().cpu()
		except:
			test_utt_data = prep_feats(sil_data[test_utt])

			if args.cuda:
				test_utt_data = test_utt_data.cuda()
				model = model.cuda()
			out_sm = model.out_proj( model.forward(test_utt_data) ).detach().cpu()

		scores.append( F.softmax(out_sm.squeeze(), dim=0)[labels_dict[speakers_enroll[i]]].item() )
		out_data.append(speakers_enroll[i]+' '+test_utt+' '+str(scores[-1]))

	print('\nScoring done')

	eer, auc, avg_precision, acc, threshold = compute_metrics(np.asarray(labels), np.asarray(scores))

	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))

	with open(args.scores_file, 'w') as f:
		for item in out_data:
			f.write("%s\n" % item)
