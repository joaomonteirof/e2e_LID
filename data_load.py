import h5py
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os
import subprocess
import shlex
from numpy.lib.stride_tricks import as_strided
import random

def augment_spec(example):

	with torch.no_grad():

		if random.random()>0.5:
			example = freq_mask(example, F=10, dim=1)
		if random.random()>0.5:
			example = freq_mask(example, F=50, dim=2)
		if random.random()>0.5:
			example += torch.randn_like(example)*random.choice([1e-1, 1e-2, 1e-3])

	return example

def freq_mask(spec, F=100, num_masks=1, replace_with_zero=False, dim=1):
	"""Frequency masking

	adapted from https://espnet.github.io/espnet/_modules/espnet/utils/spec_augment.html

	:param torch.Tensor spec: input tensor with shape (T, dim)
	:param int F: maximum width of each mask
	:param int num_masks: number of masks
	:param bool replace_with_zero: if True, masked parts will be filled with 0,
		if False, filled with mean
	:param int dim: 1 or 2 indicating to which axis the mask corresponds
	"""

	assert dim==1 or dim==2, 'Only 1 or 2 are valid values for dim!'

	F = min(F, spec.size(dim))

	with torch.no_grad():

		cloned = spec.clone()
		num_bins = cloned.shape[dim]

		for i in range(0, num_masks):
			f = random.randrange(0, F)
			f_zero = random.randrange(0, num_bins - f)

			# avoids randrange error if values are equal and range is empty
			if f_zero == f_zero + f:
				return cloned

			mask_end = random.randrange(f_zero, f_zero + f)
			if replace_with_zero:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = 0.0
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = 0.0
			else:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = cloned.mean()
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = cloned.mean()

	return cloned

def strided_app(a, L, S):
	nrows = ( (len(a)-L) // S ) + 1
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

class Loader(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100, augment=False):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)
		self.augment = augment

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		index = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		speaker_1 = self.speakers_list[index]
		spk1_utt_list = self.spk2utt[speaker_1]
		idx1, idx2 = np.random.choice(np.arange(len(spk1_utt_list)), replace=True, size=2)

		utt_1 = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx1]] )

		utt_p = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx2]] )

		neg_speaker_idx = index

		while neg_speaker_idx == index:
			neg_speaker_idx = np.random.randint(len(self.speakers_list))

		neg_speaker = self.speakers_list[neg_speaker_idx]
		nspk_utt_list = self.spk2utt[neg_speaker]

		n_idx = np.random.randint(len(nspk_utt_list))
		utt_n = self.prep_utterance( self.open_file[neg_speaker][nspk_utt_list[n_idx]] )

		return utt_1, utt_p, utt_n

	def __len__(self):
		return len(self.speakers_list)*self.n_cycles

	def prep_utterance(self, data):

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()

		if self.augment:
			data_ = augment_spec(data_)

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.speakers_list = list(open_file)

		self.n_speakers = len(self.speakers_list)

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()

class Loader_softmax(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100, augment=False):
		super(Loader_softmax, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)
		self.augment = augment

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		index = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		speaker_1 = self.speakers_list[index]
		spk1_utt_list = self.spk2utt[speaker_1]
		idx1, idx2 = np.random.choice(np.arange(len(spk1_utt_list)), replace=True, size=2)

		utt_1 = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx1]] )

		utt_p = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx2]] )

		neg_speaker_idx = index

		while neg_speaker_idx == index:
			neg_speaker_idx = np.random.randint(len(self.speakers_list))

		neg_speaker = self.speakers_list[neg_speaker_idx]
		nspk_utt_list = self.spk2utt[neg_speaker]

		n_idx = np.random.randint(len(nspk_utt_list))
		utt_n = self.prep_utterance( self.open_file[neg_speaker][nspk_utt_list[n_idx]] )

		return utt_1, utt_p, utt_n, torch.LongTensor([index])

	def __len__(self):
		return len(self.speakers_list)*self.n_cycles

	def prep_utterance(self, data):

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()

		if self.augment:
			data_ = augment_spec(data_)

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.speakers_list = list(open_file)

		self.n_speakers = len(self.speakers_list)

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()

class Loader_mining(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100, examples_per_speaker=5, augment=False):
		super(Loader_mining, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)
		self.examples_per_speaker = int(examples_per_speaker)
		self.augment = augment

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		speaker_idx = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utterances = []

		speaker = self.speakers_list[speaker_idx]
		utt_list = self.spk2utt[speaker]

		for i in range(self.examples_per_speaker):
			idx = np.random.randint(len(utt_list))
			utt = self.prep_utterance( self.open_file[speaker][utt_list[idx]] )
			utterances.append( utt )

		return torch.cat(utterances, 0).unsqueeze(1), torch.LongTensor(self.examples_per_speaker*[speaker_idx])

	def __len__(self):
		return len(self.speakers_list)*self.n_cycles

	def prep_utterance(self, data):

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()

		if self.augment:
			data_ = augment_spec(data_)

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.speakers_list = list(open_file)

		self.n_speakers = len(self.speakers_list)

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()

class Loader_pretrain(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100, augment=False):
		super(Loader_pretrain, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)
		self.augment = augment

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		speaker_idx = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		speaker = self.speakers_list[speaker_idx]
		utt_list = self.spk2utt[speaker]

		idx = np.random.randint(len(utt_list))
		utt = self.prep_utterance( self.open_file[speaker][utt_list[idx]] )

		return utt, torch.LongTensor([speaker_idx])

	def __len__(self):
		return len(self.speakers_list)*self.n_cycles

	def prep_utterance(self, data):

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()

		if self.augment:
			data_ = augment_spec(data_)

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.speakers_list = list(open_file)

		self.n_speakers = len(self.speakers_list)

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()
