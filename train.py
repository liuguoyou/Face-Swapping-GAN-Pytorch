import time
import numpy as np
import argparse
import cv2

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import model
import dataset
import criterion


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--img_size', type=int, default=128)
	parser.add_argument('--img_path', type=str, default='')
	parser.add_argument('--seg_path', type=str, default='')
	parser.add_argument('--ckpt_path', type=str, default='')
	parser.add_argument('--vgg_path', type=str, default='')
	# parser.add_argument('--pkl_path', type=str, default='/home/yy/FSGAN/data_full/viddata.pkl')
	# parser.add_argument('--pkl_path', type=str, default='/home/yy/FSGAN/Allen_KangHui/src_aligned/src_landmark.txt')

	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--batch_size', type=int, default=10) # 32 for 128, 10 for 256
	parser.add_argument('--max_iter', type=int, default=1920000)

	parser.add_argument('--con_iter', type=int, default=3)

	# parser.add_argument('--start_batch', type=int, default=3773)
	args = parser.parse_args()

	writer = SummaryWriter()

	gpus = (0, 1, 2, 3)
	reenactor = model.Reenactment()
	generator = model.Generator(reenactor, FLoss=False, resume_path=args.vgg_path, consistensy_iter=args.con_iter)
	generator = nn.DataParallel(generator, device_ids=gpus).cuda()

	multidis = model.MultiscaleDiscriminator(71)
	lossfunction_dis = criterion.LSGanLoss()
	discriminator = model.Discriminator(multidis, lossfunction_dis)
	discriminator = nn.DataParallel(discriminator, device_ids=gpus).cuda()

	if args.resume:
		g_ckpt = torch.load(args.ckpt_path + '/generator_final.pth.tar')
		d_ckpt = torch.load(args.ckpt_path + '/discriminator_final.pth.tar')
		generator.module.model.load_state_dict(g_ckpt)
		discriminator.module.load_state_dict(d_ckpt)

	trainset = dataset.Reenactset(
		pkl_path=args.pkl_path,
		img_path=args.img_path,
		seg_path=args.seg_path,
		max_iter=args.max_iter,
		consistency_iter = args.con_iter,
		image_size=args.img_size
		)

	trainloader = Data.DataLoader(
		trainset,
		batch_size=args.batch_size*len(gpus),
		shuffle=False,
		num_workers=len(gpus) * 8
		)

	num_batches = len(trainloader)

	print('Iters: %d'%num_batches)
	# assert num_batches == (args.max_iter // args.batch_size // len(gpus))

	optimizerG = torch.optim.Adam(
		generator.module.model.parameters(),
		lr=2e-4,
		betas=(0.5, 0.999)
		)
	optimizerD = torch.optim.Adam(
		discriminator.module.parameters(),
		lr=2e-4,
		betas=(0.5, 0.999)
		)

	generator.train()
	discriminator.train()
	start = time.time()
	for batch_idx, (inputs, target_hmap, target, target_seg, m_hmaps) in enumerate(trainloader):
		inputs = inputs.cuda()
		target_hmap = target_hmap.cuda()
		target = target.cuda()
		target_seg = target_seg.cuda()

		output, source_seg, loss_gen, con_out = generator(inputs, target, target_seg, target_hmap, m_hmaps)

		_, _, loss_gen_D, _ = discriminator(output, target, target_hmap)
		loss_gen = loss_gen.mean() + loss_gen_D.mean() * 0.001
		_, _, _, loss_dis = discriminator(output.detach(), target, target_hmap)

		# 128
		if batch_idx <= (3 * num_batches / 4):
			set_lr = 2e-4 - (2e-4 - 1e-5) * batch_idx * 4 / num_batches / 3
			for param_group in optimizerG.param_groups:
				param_group['lr'] = set_lr
			for param_group in optimizerD.param_groups:
				param_group['lr'] = set_lr

		# # 256
		# if batch_idx <= (1 * num_batches / 4):
		# 	set_lr = 1e-4 - (1e-4 - 1e-5) * batch_idx * 4 / num_batches
		# 	for param_group in optimizerG.param_groups:
		# 		param_group['lr'] = set_lr
		# 	for param_group in optimizerD.param_groups:
		# 		param_group['lr'] = set_lr
		# elif (batch_idx <= (3 * num_batches / 4)) and batch_idx > (1 * num_batches / 4):
		# 	set_lr = 1e-5 - (1e-5 - 1e-6) * (batch_idx * 2 / num_batches - 0.5)
		# 	for param_group in optimizerG.param_groups:
		# 		param_group['lr'] = set_lr
		# 	for param_group in optimizerD.param_groups:
		# 		param_group['lr'] = set_lr

		optimizerG.zero_grad()
		loss_gen.backward()
		optimizerG.step()

		loss_dis = torch.mean(loss_dis)
		optimizerD.zero_grad()
		loss_dis.backward()
		optimizerD.step()

		lr = optimizerG.param_groups[0]['lr']

		speed = (time.time() - start) / (batch_idx + 1)
		remain_time = (num_batches - batch_idx - 1) * speed / 3600

		if (batch_idx % 500 == 0):
			for i in range(10):
				cv2.imwrite('work/vis/%d_%d_inputs.jpg'%(batch_idx, i), inputs.cpu().detach().numpy()[i, 0:3, :, :].transpose(1,2,0)*255)
				cv2.imwrite('work/vis/%d_%d_target.jpg'%(batch_idx, i), target.cpu().detach().numpy()[i, :, :, :].transpose(1,2,0)*255)
				cv2.imwrite('work/vis/%d_%d_output.jpg'%(batch_idx, i), output.cpu().detach().numpy()[i, :, :, :].transpose(1,2,0)*255)
				cv2.imwrite('work/vis/%d_%d_conout.jpg'%(batch_idx, i), con_out.cpu().detach().numpy()[i, :, :, :].transpose(1,2,0)*255)
				cv2.imwrite('work/vis/%d_%d_segment.jpg'%(batch_idx, i), source_seg.cpu().detach().numpy()[i, :, :, :].transpose(1,2,0)*255)
				# cv2.imwrite('work/vis/%d_%d_heatmap.jpg'%(batch_idx, i), np.sum(target_hmap.cpu().detach().numpy()[i, :, :, :].transpose(1,2,0)*255, 2))

		print('Batch: %d/%d GLoss: %f DLoss: %f LR: %f Speed: %.2f Remaining time: %.2f hrs'%(
			batch_idx,
			num_batches,
			loss_gen,
			loss_dis,
			lr,
			speed,
			remain_time
			))

		if (batch_idx % 100 == 0):
			writer.add_images('target', target[0:4, :, :, :], 0)
			writer.add_images('output', output[0:4, :, :, :], 0)
			writer.add_images('segment', source_seg[0:4, :, :, :], 0)
		writer.add_scalars('Log', {'Gen': loss_gen, 'Dis': loss_dis}, batch_idx)
		writer.add_scalar('LR', lr, batch_idx)

		if (batch_idx % 5000 == 0) and (batch_idx > 0):
			torch.save(generator.module.model.state_dict(), 'work/generator_%02d.pth.tar'%(batch_idx))
			torch.save(discriminator.module.state_dict(), 'work/discriminator_%02d.pth.tar'%(batch_idx))

	torch.save(generator.module.model.state_dict(), 'work/generator_final.pth.tar')
	torch.save(discriminator.module.state_dict(), 'work/discriminator_final.pth.tar')

	writer.close()
