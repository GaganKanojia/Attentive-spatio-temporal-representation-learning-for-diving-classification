import numpy as np
from torch.utils.data import Dataset
import random
import cv2 as cv
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DeepSeqDataset(Dataset):

	def __init__(self, frames_dir, vocab, data_48,seq_len,phase,transform=None):
		self.frames_dir = frames_dir
		self.vocab = vocab
		self.data_48 = data_48
		self.seq_len = seq_len
		self.phase = phase

	def __len__(self):
		return len(self.data_48)

	def __getitem__(self, idx):
		file_ = self.data_48[idx]
		folder_name = file_['vid_name']
		label = file_['label']
		action = self.vocab[label,:].astype(np.int32)
		
		print(folder_name)
		folder_path = os.path.join(self.frames_dir,folder_name)

		revert= np.array(range(self.seq_len-1,-1,-1))
		vid_frames = np.zeros((self.seq_len,3,224,224),dtype=np.float32)
		
		num_images = len(os.listdir(folder_path))
		a = np.random.uniform(0,0.99999,self.seq_len)
		im_indices = np.sort((a*num_images).astype(np.int))

		random_crop_r = random.randint(-10,10)
		random_crop_c = random.randint(-10,10)
		for j in range(0,self.seq_len):

			im_path = folder_path+'/frame{:04d}.jpg'.format(im_indices[j])
			im = Image.open(im_path)

			cols,rows = im.size
			min_edge_size = np.min([rows,cols])

			if self.phase == 'train':
				scale = 245.0/min_edge_size			
				im = im.resize((int(cols*scale), int(rows*scale)), Image.ANTIALIAS)
				cols,rows = im.size
				row_start = np.int((rows-224)/2)+random_crop_r
				cols_start = np.int((cols-224)/2)+random_crop_c
			elif self.phase == 'test':
				scale = 224.0/min_edge_size			
				im = im.resize((int(cols*scale), int(rows*scale)), Image.ANTIALIAS)
				cols,rows = im.size
				row_start = np.int((rows-224)/2)
				cols_start = np.int((cols-224)/2)

			im = np.array(im)
			im_crop= im[row_start:row_start+224,cols_start:cols_start+224,:]
			
			im = (np.rollaxis(im_crop,2)) /255
			
			im[0,:,:] = (im[0,:,:]-0.485)/0.229
			im[1,:,:] = (im[1,:,:]-0.456)/0.224
			im[2,:,:] = (im[2,:,:]-0.406)/0.225
			vid_frames[j,:,:,:] = im

		vid_frames_revert = vid_frames[revert,:,:,:]
		sample = {'vid_frames': vid_frames,'vid_frames_revert': vid_frames_revert, 'label': label, 'action': action}

		return sample


