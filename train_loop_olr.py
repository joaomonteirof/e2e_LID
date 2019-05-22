import torch
import torch.nn.functional as F

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

import torchvision.transforms as transforms
from PIL import ImageFilter
from utils.harvester import TripletHarvester

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, patience, checkpoint_path=None, checkpoint_epoch=None, swap=False, softmax=False, mining=False, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.softmax = softmax!='none'
		self.mining = mining
		self.model = model
		self.swap = swap
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.harvester = TripletHarvester()

		if self.valid_loader is not None:
			self.history = {'train_loss': [], 'train_loss_batch': [], 'valid_loss': []}
			self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience, verbose=True, threshold=1e-4, min_lr=1e-6)
		else:
			self.history = {'train_loss': [], 'train_loss_batch': []}

			if checkpoint_epoch is not None:
				self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 50, 100], gamma=0.5, last_epoch=checkpoint_epoch)
			else:
				self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 50, 100], gamma=0.5)

		if self.softmax:
			self.history['softmax_batch']=[]
			self.history['softmax']=[]

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			np.random.seed()
			train_iter = tqdm(enumerate(self.train_loader))
			train_loss_epoch=0.0
			gtr_epoch=0.0

			if not self.softmax:

				for t, batch in train_iter:
					train_loss = self.train_step(batch)
					self.history['train_loss_batch'].append(train_loss)
					train_loss_epoch+=train_loss
					self.total_iters += 1

				self.history['train_loss'].append(train_loss_epoch/(t+1))

				print('Total train loss, {:0.4f}'.format(self.history['train_loss'][-1]))

			else:
				ce_epoch=0.0
				for t, batch in train_iter:
					train_loss, ce = self.train_step(batch)
					self.history['train_loss_batch'].append(train_loss)
					self.history['softmax_batch'].append(ce)
					train_loss_epoch+=train_loss
					ce_epoch+=ce
					self.total_iters += 1

				self.history['train_loss'].append(train_loss_epoch/(t+1))
				self.history['softmax'].append(ce_epoch/(t+1))

				print('Total train loss, Triplet loss, and Cross-entropy: {:0.4f}, {:0.4f}, {:0.4f}'.format(self.history['train_loss'][-1], (self.history['train_loss'][-1]-self.history['softmax'][-1]), self.history['softmax'][-1]))

			if self.valid_loader is not None:
				val_loss = 0.0
				for t, batch in enumerate(self.valid_loader):
					val_loss+=self.valid(batch)
				self.history['valid_loss'].append(val_loss/(t+1))

				print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

				self.scheduler.step(self.history['valid_loss'][-1])

			else:
				self.scheduler.step()

			self.cur_epoch += 1


			if self.cur_epoch % save_every == 0 or self.history['valid_loss'][-1] < np.min([np.inf]+self.history['valid_loss'][:-1]):
				self.checkpointing()

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		if self.mining:
			utterances, y = batch
			utterances.resize_(utterances.size(0)*utterances.size(1), utterances.size(2), utterances.size(3), utterances.size(4))
			y.resize_(y.numel())
		elif self.softmax:
			utt_a, utt_p, utt_n, y = batch
		else:
			utt_a, utt_p, utt_n = batch

		if self.mining:

			ridx = np.random.randint(utterances.size(3)//2, utterances.size(3))
			utterances = utterances[:,:,:,:ridx]
			if self.cuda_mode:
				utterances = utterances.cuda()

			embeddings = self.model.forward(utterances)
			embeddings_norm = torch.div(embeddings, torch.norm(embeddings, 2, 1).unsqueeze(1).expand_as(embeddings))

			triplets_idx = self.harvester.harvest_triplets(embeddings_norm.detach().cpu(), y.numpy())

			if self.cuda_mode:
				triplets_idx = triplets_idx.cuda()

			emb_a = torch.index_select(embeddings, 0, triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings, 0, triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings, 0, triplets_idx[:, 2])

		else:
			ridx = np.random.randint(utt_a.size(3)//2, utt_a.size(3))
			utt_a, utt_p, utt_n = utt_a[:,:,:,:ridx], utt_p[:,:,:,:ridx], utt_n[:,:,:,:ridx]

			if self.cuda_mode:
				utt_a, utt_p, utt_n = utt_a.cuda(), utt_p.cuda(), utt_n.cuda()

			emb_a, emb_p, emb_n = self.model.forward(utt_a), self.model.forward(utt_p), self.model.forward(utt_n)
			embeddings_norm = torch.div(emb_a, torch.norm(emb_a, 2, 1).unsqueeze(1).expand_as(emb_a))

		loss = self.triplet_loss(emb_a, emb_p, emb_n, swap=self.swap)

		if self.softmax:
			if self.cuda_mode:
				y = y.cuda().squeeze()

			ce = F.cross_entropy(self.model.out_proj(embeddings_norm,y), y)
			loss += ce
			loss.backward()
			self.optimizer.step()
			return loss.item(), ce.item()
		else:
			loss.backward()
			self.optimizer.step()
			return loss.item()

	def valid(self, batch):

		self.model.eval()

		xa, xp, xn, y = batch
		y = y.squeeze()

		ridx = np.random.randint(xa.size(3)//2, xa.size(3))

		xa = xa[:,:,:,:ridx]

		if self.cuda_mode:
			xa = xa.contiguous().cuda()
			y = y.cuda()

		emb_a = self.model.forward(xa)

		output = self.model.out_proj(emb_a)

		pred = torch.argmax(output, dim=1)
		correct = pred.eq(y).sum().item()

		return 1.-correct/xa.size(0)

	def triplet_loss(self, emba, embp, embn, swap, reduce_=True):

		d_ap = 1.-F.cosine_similarity(emba, embp)
		d_an = 1.-F.cosine_similarity(emba, embn)

		if swap:

			d_pn = 1.-F.cosine_similarity(embp, embn)

			loss_ = F.softplus(d_ap - torch.min(d_an, d_pn))

		else:
			loss_ = F.softplus(d_ap - d_an)

		return loss_.mean() if reduce_ else loss_

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda()

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))
