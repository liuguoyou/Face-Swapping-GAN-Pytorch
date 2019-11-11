import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
import math


Norm = nn.InstanceNorm2d
# Norm = nn.BatchNorm2d

# Generator

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
	return nn.Conv2d(
		in_planes,
		out_planes,
		kernel_size=3,
		stride=stride,
		padding=dilation,
		bias=False,
		dilation=dilation
		)

class Inputlayer(nn.Module):
	def __init__(self, in_planes, out_planes=128, kernel_size=7, stride=1, padding=3):
		super(Inputlayer, self).__init__()
		self.conv1 = nn.Conv2d(
			in_planes,
			out_planes,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			bias=False,
			dilation=1
			)
		self.bn1 = Norm(out_planes)
		self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		return x

class Bottleneck(nn.Module):
	def __init__(self, in_planes, out_planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = conv3x3(in_planes, in_planes, stride)
		self.bn1 = Norm(in_planes)
		self.conv2 = conv3x3(in_planes, out_planes)
		self.bn2 = Norm(out_planes)
		self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out = x + out
		out = self.relu(out)

		return out

class Downsample(nn.Module):
	def __init__(self, in_planes, out_planes, stride=2):
		super(Downsample, self).__init__()
		self.conv1 = conv3x3(in_planes, out_planes, stride)
		self.bn1 = Norm(out_planes)
		self.conv2 = conv3x3(out_planes, out_planes)
		self.bn2 = Norm(out_planes)
		self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		return x

class Upsample(nn.Module):
	def __init__(self, in_planes, out_planes, stride=1, scale_factor=2):
		super(Upsample, self).__init__()
		self.scale_factor = scale_factor
		self.conv1 = conv3x3(in_planes, out_planes, stride)
		self.bn1 = Norm(out_planes)
		self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		return x

class Outputlayer(nn.Module):
	def __init__(self, in_planes=128, out_planes=3, kernel_size=7, stride=1, padding=3):
		super(Outputlayer, self).__init__()
		self.conv1 = nn.Conv2d(
			in_planes,
			out_planes,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			bias=False,
			dilation=1
			)
		self.bn1 = Norm(out_planes)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)

		x = self.tanh(x)

		return x

class Globalgenerator(nn.Module):
	def __init__(self, n1, n2, n3, n4, in_planes, out_planes):
		super(Globalgenerator, self).__init__()
		self.inputlayer = Inputlayer(in_planes=in_planes)

		self.down1 = Downsample(128, 256)
		self.down2 = Downsample(256, 512)
		self.down3 = Downsample(512, 1024)

		self.bottleneck1 = self._make_layer(n1, 128)
		self.bottleneck2 = self._make_layer(n2, 256)
		self.bottleneck3 = self._make_layer(n3, 512)
		# self.bottleneck4 = self._make_layer(n4, 1024)

		self.up1 = Upsample(256, 128)
		self.up2 = Upsample(512, 256)
		self.up3 = Upsample(1024, 512)

		self.outputlayer = Outputlayer(out_planes=out_planes)

	def _make_layer(self, n, planes):
		layers = []
		for i in range(n):
			layers.append(Bottleneck(planes, planes))
		layers = nn.Sequential(*layers)

		return layers

	def forward(self, x):
		x = self.inputlayer(x)

		x1 = self.down1(x)
		x2 = self.down2(x1)
		# x3 = self.down3(x2)

		# x4 = self.bottleneck4(x3)

		# x3 = x3 + x4
		# x3 = self.up3(x3)
		x3 = self.bottleneck3(x2)

		x2 = x2 + x3
		x2 = self.up2(x2)
		x2 = self.bottleneck2(x2)

		x1 = x1 + x2
		x1 = self.up1(x1)
		x1 = self.bottleneck1(x1)

		x = self.outputlayer(x1)

		return x

class Enhancer(nn.Module):
	def __init__(self, generator, n, scale_factor, in_planes, out_planes):
		super(Enhancer, self).__init__()
		self.inputlayer = Inputlayer(in_planes=in_planes)
		self.down = Downsample(128, 256)

		self.up1 = Upsample(256, 128)
		self.bottleneck1 = self._make_layer(n, 128)
		self.outputlayer = Outputlayer(out_planes=out_planes)

		self.up2 = Upsample(256, 128)
		self.bottleneck2 = self._make_layer(n, 128)
		self.seglayer = nn.Sequential(
			nn.Conv2d(128, 1, kernel_size=1),
			nn.Sigmoid()
			)

		self.generator = generator
		self.scale_factor = scale_factor

	def _make_layer(self, n, planes):
		layers = []
		for i in range(n):
			layers.append(Bottleneck(planes, planes))
		layers = nn.Sequential(*layers)

		return layers

	def forward(self, x):
		x1 = self.inputlayer(x)
		x1 = self.down(x1)

		x2 = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
		x2 = self.generator(x2)

		feature = x1 + x2
		x = self.up1(feature)
		x = self.bottleneck1(x)
		outputs = self.outputlayer(x)

		s = self.up2(feature)
		s = self.bottleneck2(s)
		segment = self.seglayer(s)

		return outputs, segment

def Reenactment(n=2, n1=2, n2=2, n3=3, n4=3):
	generator = Globalgenerator(n1, n2, n3, n4, 71, 256)
	reenactor = Enhancer(generator, n, 0.5, 71, 3)
	return reenactor

# Discriminator

class MultiscaleDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
		super(MultiscaleDiscriminator, self).__init__()
		self.num_D = num_D
		self.n_layers = n_layers

		for i in range(num_D):
			netD = NLayerDiscriminator(input_nc, ndf, n_layers)
			setattr(self, 'layer'+str(i), netD.model)

		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

	def singleD_forward(self, model, input):
		return model(input)

	def forward(self, input):
		num_D = self.num_D
		result = []
		input_downsampled = input
		for i in range(num_D):
			model = getattr(self, 'layer'+str(num_D-1-i))
			result.append(self.singleD_forward(model, input_downsampled))
			if i != (num_D-1):
				input_downsampled = self.downsample(input_downsampled)
		return result

class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3):
		super(NLayerDiscriminator, self).__init__()
		self.n_layers = n_layers

		kw = 4
		padw = int(np.ceil((kw-1.0)/2))
		sequence = [[
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.1, True)
			]]

		nf = ndf
		for n in range(1, n_layers):
			nf_prev = nf
			nf = min(nf*2, 512)
			sequence += [[
				nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
				Norm(nf),
				nn.LeakyReLU(0.1, True)
			]]

		nf_prev = nf
		nf = min(nf*2, 512)
		sequence += [[
			nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
			Norm(nf),
			nn.LeakyReLU(0.1, True)
		]]

		sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

		sequence_stream = []
		for n in range(len(sequence)):
			sequence_stream += sequence[n]
		self.model = nn.Sequential(*sequence_stream)

	def forward(self, input):
		return self.model(input)

# Utils

class PerceptualModel(torch.nn.Module):
	def __init__(self, resume_path=''):
		super(PerceptualModel, self).__init__()
		vgg_pretrained_model = torchvision.models.vgg19_bn(pretrained=False, num_classes=47060)
		vgg_pretrained_model.load_state_dict(torch.load(resume_path))
		vgg_pretrained_features = vgg_pretrained_model.features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		for x in range(6):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(6, 13):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(13, 26):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(26, 39):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		h = self.slice1(x)
		h_relu1_2 = h
		h = self.slice2(h)
		h_relu2_2 = h
		h = self.slice3(h)
		h_relu3_4 = h
		h = self.slice4(h)
		h_relu4_4 = h
		return h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4

class Generator(nn.Module):
	def __init__(self, model, FLoss=False, resume_path='', consistensy_iter=2):
		super(Generator, self).__init__()
		self.FLoss = FLoss
		self.consistensy_iter = consistensy_iter
		self.model = model
		for m in self.model.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Perceptual model
		self.p_model = PerceptualModel(resume_path)
		self.p_model.eval()
		# kernel_size = 65
		# sigma = 4
		# x_cord = torch.arange(kernel_size)
		# x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
		# y_grid = x_grid.t()
		# xy_grid = torch.stack([x_grid, y_grid], dim=-1)
		# mean = (kernel_size - 1)/2.
		# variance = sigma**2.
		# gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
		# gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

		# gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
		# gaussian_kernel = gaussian_kernel.repeat(68, 1, 1, 1)

		# self.gaussian_filter = nn.Conv2d(in_channels=68, out_channels=68,
		# 	kernel_size=kernel_size, groups=68, bias=False)
		# self.gaussian_filter.weight.data = gaussian_kernel
		# self.gaussian_filter.weight.requires_grad = False

	def forward(self, source_img, target_img, target_seg, hmap, m_hmaps):

		# position = torch.cat([torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.arange(68, dtype=torch.float32), 1), 0), 0)]*source_img.shape[0])
		# position = torch.cat((position.cuda(), pts), 3)
		# position = position.permute(0, 1, 3, 2).long()
		# hmap = torch.zeros((position.shape[0], 68, 320, 320)).cuda()
		# for n in range(position.shape[0]):
		# 	hmap[n][position[n][0][0], position[n][0][2]+32, position[n][0][1]+32]=1
		# hmap = self.gaussian_filter(hmap)

		inputs = torch.cat((source_img, hmap), 1)
		outputs, segment = self.model(inputs)
		out_seg = torch.mul(outputs, segment)
		tar_seg = torch.mul(target_img, target_seg)

		pix_loss = torch.mean(torch.abs(out_seg - tar_seg))
		# pis_loss = 0.1 * torch.mul(torch.abs(outputs - target_img), target_seg).sum() / target_seg.sum()

		# if len(m_hmaps) > 0:
		m_img = source_img
		for m_hmap in m_hmaps:
			m_hmap = m_hmap.cuda()
			m_img, _ = self.model(torch.cat((m_img, m_hmap), 1))
		con_out, con_seg = self.model(torch.cat((m_img, hmap), 1))
		con_out_seg = torch.mul(con_out, con_seg)
		pix_loss = torch.mean(torch.abs(con_out_seg - tar_seg)) + pix_loss
		# pis_loss = 0.1 * torch.mul(torch.abs(con_out - target_img), target_seg).sum() / target_seg.sum() + pix_loss

		if self.FLoss:
			BCE_loss = F.binary_cross_entropy_with_logits(segment, torch.unsqueeze(target_seg[:, 0, :, :], 1), reduction='none')
			pt = torch.exp(-BCE_loss)
			seg_loss = alpha * (1 - pt) ** gamma * BCE_loss
		else:
			seg_loss = F.binary_cross_entropy(segment, torch.unsqueeze(target_seg[:, 0, :, :], 1)) * 0.1

		# Perceptual model

		with torch.no_grad():
			fea_1, fea_2, fea_3, fea_4 = self.p_model(out_seg)
			tar_fea_1, tar_fea_2, tar_fea_3, tar_fea_4 = self.p_model(tar_seg)
			con_fea_1, con_fea_2, con_fea_3, con_fea_4 = self.p_model(con_out_seg)

		per_loss = torch.mean(torch.abs(fea_1 - tar_fea_1)) + \
		torch.mean(torch.abs(fea_2 - tar_fea_2)) + \
		torch.mean(torch.abs(fea_3 - tar_fea_3)) + \
		torch.mean(torch.abs(fea_4 - tar_fea_4)) + \
		torch.mean(torch.abs(con_fea_1 - tar_fea_1)) + \
		torch.mean(torch.abs(con_fea_2 - tar_fea_2)) + \
		torch.mean(torch.abs(con_fea_3 - tar_fea_3)) + \
		torch.mean(torch.abs(con_fea_4 - tar_fea_4))

		gen_loss = (seg_loss + pix_loss) * 0.1 + per_loss

		return outputs, segment, gen_loss, con_out

class Discriminator(nn.Module):
	def __init__(self, model, lossfunction):
		super(Discriminator, self).__init__()
		self.model = model
		for m in self.model.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.lossfunction = lossfunction

	def forward(self, outputs, labels, hmap):
		fake = self.model(torch.cat((outputs, hmap), 1))
		real = self.model(torch.cat((labels, hmap), 1))
		loss_G, loss_D = self.lossfunction(real, fake)
		return fake, real, loss_G, loss_D
