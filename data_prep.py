import argparse
import h5py
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Change path in feats.scp')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	args = parser.parse_args()

	if os.path.isfile(args.out_path+args.out_name):
		os.remove(args.out_path+args.out_name)
		print(args.out_path+args.out_name+' Removed')

	scp_list = glob.glob(args.path_to_data + '*.scp')

	if len(scp_list)<1:
		print('Nothing found at {}.'.format(args.path_to_data))
		exit(1)

	print('Start of data preparation')

	hdf = h5py.File(args.out_path+args.out_name, 'a')

	for file_ in scp_list:

		print('Processing file {}'.format(file_))

		data = { k:m for k,m in read_mat_scp(file_) }

		for i, utt in enumerate(data):

			print('Storing utterance ' + utt)

			data_ = data[utt]
			#data_ = ( data_ - data_.mean(0) ) / data_.std(0)
			features = data_.T

			if features.shape[0]>0:
				features = np.expand_dims(features, 0)
				hdf.create_dataset(utt, data=features, maxshape=(features.shape[0], features.shape[1], features.shape[2]))
			else:
				print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt))

	hdf.close()
