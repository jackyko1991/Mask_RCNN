import SimpleITK as sitk
import os
import pandas as pd
import random
import tensorflow as tf
import numpy as np
from multiprocessing import Pool

def label_to_name(label_df, label):
	return label_df.loc[label_df['Index'] == label]['Finding Label'].values[0]

def name_to_label(label_df, name):
	return label_df.loc[label_df['Finding Label'] == name]['Index'].values[0]

class Dataset(object):
	"""
	  Load image-label pair for training, testing and inference.
	  Here we implement SimpleITK for image io.
	  Currently only support linear interpolation method
	  Args:
		data_dir (string): Path to data directory.
	    transforms (list): List of SimpleITK image transformations.
	    train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
	"""

	def __init__(self,
		data_dir = '',
		transforms=None,
		train=False,
		image_format="png",
		bbox_csv='',
		data_entry_csv='',
		no_finding_prob=0.1,
		label_csv=None):

		# Init membership variables
		self.data_dir = data_dir
		self.transforms = transforms
		self.train = train
		self.image_format = image_format
		self.bbox_csv = bbox_csv
		self.data_entry_csv = data_entry_csv
		self.bbox = None
		self.data_entry = None
		self.no_finding_prob = no_finding_prob
		self.label_csv = label_csv
		self.label_df = None

	def drop(self,probability):
		return random.random() <= probability 

	def get_dataset(self):
		# load bbox and data_entry csv files
		self.bbox = pd.read_csv(self.bbox_csv)
		self.data_entry = pd.read_csv(self.data_entry_csv)

		# load label csv file
		self.label_df = pd.read_csv(self.label_csv)

		# create list of images
		print("Generating list of images with bbox...")
		image_paths = os.listdir(self.data_dir)

		# extract images with bbox or no findings
		bbox_image_paths = []
		pool = Pool()
		bbox_image_paths = pool.map(self.bbox_list_worker, image_paths)
		pool.close()
		pool.join()

		print("Generate list of images with bbox complete")

		bbox_image_paths = list(filter(None,bbox_image_paths))

		dataset = tf.data.Dataset.from_tensor_slices((bbox_image_paths))
		dataset = dataset.map(lambda image_filename: tuple(tf.py_func(
		self.input_parser, [image_filename], [tf.string, tf.float32,tf.uint8, tf.float32])))

		self.dataset = dataset
		self.data_size = len(image_paths)
		return self.dataset

	def read_image(self,filename):
		reader = sitk.ImageFileReader()
		reader.SetFileName(os.path.join(self.data_dir,filename))
		img = reader.Execute()

		# provide image spacing if input file is png
		if self.image_format == "png":
			entry = self.data_entry.loc[self.data_entry['Image Index'] == filename]

			# calculate the world space width and height
			width_world = entry['OriginalWidth'].values[0] * entry['OriginalImagePixelSpacingx'].values[0]
			height_world = entry['OriginalHeight'].values[0] * entry['OriginalImagePixelSpacingy'].values[0]

			if width_world > height_world:
				img.SetSpacing((width_world/1024.,width_world/1024.))
			else:
				img.SetSpacing((height_world/1024.,height_world/1024.))

		return img

	def bbox_list_worker(self,image_filename):
		# check if image has a bbox label
		if image_filename in self.bbox["Image Index"].tolist():
			return image_filename
		else:
			# has some probability to use no finding image
			entry = self.data_entry.loc[self.data_entry['Image Index'] == image_filename]

			if ("No Finding" in entry["Finding Labels"].tolist()) and self.drop(self.no_finding_prob):
				return image_filename
			else:
				return None

	def read_label(self,filename):
		entry = self.bbox.loc[self.bbox['Image Index'] == filename]
		findings = []
		bboxes = []

		if len(entry.index) > 0:
			for i in range(len(entry.index)):
				findings.append(name_to_label(self.label_df,entry['Finding Label'].values[i]))
				bboxes.append([entry['x'].values[i],
							entry['y'].values[i],
							entry['w'].values[i],
							entry['h'].values[i]])
		else:
			findings.append(0)
			bboxes.append([0,0,0,0])

		return findings, bboxes

	def input_parser(self,image_filename):
		# read image and label
		image = self.read_image(image_filename.decode("utf-8"))
		findings, bboxes = self.read_label(image_filename.decode("utf-8"))

		sample = {'image':image, 'findings': findings, 'bboxes': bboxes}

		if self.transforms:
			for transform in self.transforms:
				sample = transform(sample)

		# convert sample to tf tensors
		image = sitk.GetArrayFromImage(sample['image'])

		# type casting
		image = np.asarray(image, np.float32)
		findings = np.asarray(sample['findings'], np.uint8)
		bboxes = np.asarray(sample['bboxes'], np.float32)

		# # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
		# image_np = np.transpose(image_np,(2,1,0))
		# label_np = np.transpose(label_np,(2,1,0))

		return image_filename.decode("utf-8"), image, findings, bboxes

class BboxNihToMrcnn(object):
	"""
	Convert bbox coordinate from nih raw data to mrcnn network
	"""

	def __init__(self):
		self.name = 'BboxNihToMrcnn'

	def __call__(self, sample):
		image = sample['image']
		findings = sample['findings']
		bboxes = sample['bboxes']

		# xywh to y1x1y2x2
		bboxes_ = []
		for bbox in bboxes:
			bbox[2] = bbox[0] + bbox[2]
			bbox[3] = bbox[1] + bbox[3]

			bbox[0], bbox[1] = np.round(bbox[1]), np.round(bbox[0])
			bbox[2], bbox[3] = np.round(bbox[3]), np.round(bbox[2])

			bboxes_.append(bbox)

		bboxes = bboxes_

		return {'image': image, 'findings': findings, 'bboxes': bboxes}