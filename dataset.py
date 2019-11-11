import numpy as np
import cv2
from glob import glob
import os
import os.path as path
import random
import pickle

import scipy.stats as st

import torch
import torch.utils.data as Data


def process_pts(line):
	line = line.replace(',', '')
	line = line.split(' ')
	fname = line[0]
	pts = line[1:-3]
	ang = line[-3:]
	ang = [float(i) for i in ang]
	ang = np.float32(ang)
	pts = [float(i) for i in pts]
	pts = np.float32(pts)
	pts = pts.reshape([-1,3])
	return fname, pts, ang

def get_hmap(pts, size=128):
	pos = np.dstack(np.mgrid[0:size:1, 0:size:1])
	hmap = np.zeros([size, size, 68])
	for i, point in enumerate(pts):
		p_resize = point / 384 * size # Please change the size range of pts(default is 384)
		hmap[:, :, i] = st.multivariate_normal(mean=[p_resize[1], p_resize[0]], cov=16).pdf(pos)
	return hmap

def Seg2map(seg, size=128, interpolation=cv2.INTER_NEAREST):
	seg_new = np.zeros(seg.shape, dtype='float32')
	seg_new[seg > 7.5] = 1
	seg = np.copy(cv2.resize(seg_new, (size, size), interpolation=interpolation))
	return seg

def Cv2tensor(img):
	img = img.transpose(2, 0, 1)
	img = torch.from_numpy(img.astype(np.float32))
	return img

class Reenactset(Data.Dataset):
	def __init__(self, pkl_path='', img_path='', seg_path='', max_iter=80000, consistency_iter=2, image_size=128):
		super(Reenactset, self).__init__()
		self.img_path = img_path
		self.seg_path = seg_path
		self.data = pickle.load(open(pkl_path, 'rb'))
		self.idx = list(self.data.keys())
		self.size = max_iter
		self.image_size = image_size
		self.consistency_iter = consistency_iter
		assert self.consistency_iter > 0

	def __getitem__(self, index):
		ID = random.choice(self.idx)
		samples = random.sample(self.data[ID], self.consistency_iter+1)
		source = samples[0]
		target = samples[1]
		mid_samples = samples[2:]

		source_name, _, _ = process_pts(source)
		target_name, pts, _ = process_pts(target)

		m_pts = []
		for m_s in mid_samples:
			_, m_pt, _ = process_pts(m_s)
			m_pts.append(m_pt)

		# pts = torch.from_numpy(pts[:, 0:2].astype(np.float32))
		# pts = torch.unsqueeze(pts, 0)
		m_hmaps = []
		hmap = Cv2tensor(get_hmap(pts, size=self.image_size))
		for m_pt in m_pts:
			m_hmaps.append(Cv2tensor(get_hmap(m_pt, size=self.image_size)))

		source_file = self.img_path + f'/img/{ID}/{source_name}'
		target_file = self.img_path + f'/img/{ID}/{target_name}'
		target_seg_file = self.seg_path + f'/seg/{ID}/seg_{target_name}'

		source_img = cv2.imread(source_file)
		source_img = cv2.resize(source_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
		source_img = source_img / 255
		source_img = Cv2tensor(source_img)

		target_img = cv2.imread(target_file)
		target_img = cv2.resize(target_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
		target_img = target_img / 255
		target_img = Cv2tensor(target_img)

		# source_seg = cv2.imread(self.img_path + f'/seg/{ID}/seg_{source_name}')
		# source_seg = Seg2map(source_seg, size=self.image_size)
		# source_seg = Cv2tensor(source_seg)

		target_seg = cv2.imread(target_seg_file)
		target_seg = Seg2map(target_seg, size=self.image_size)
		target_seg = Cv2tensor(target_seg)

		return source_img, hmap, target_img, target_seg, m_hmaps

	def __len__(self):
		return self.size
