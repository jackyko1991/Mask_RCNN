import os
import Config
import datetime
import tensorflow as tf
import sys
import dataset
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from mrcnn import visualize

class Trainer:
	def __init__(self):
		self.config = None
		self.trainDataset = None
		self.testDataset = None

	def train(self):
		# load data
		train_iterator = self.trainDataset.make_initializable_iterator()
		next_element_train = train_iterator.get_next()

		label_df = pd.read_csv("./label.csv")


		# training cycle
		with tf.Session() as sess:
			# Initialize all variables
			sess.run(tf.global_variables_initializer())
			print("{}: Start training...".format(datetime.datetime.now()))

			for epoch in range(self.config.getConfigValue("EPOCHS")):
				# initialize iterator in each new epoch
				sess.run(train_iterator.initializer)

				print("{}: Epoch {} starts".format(datetime.datetime.now(),epoch+1))

				# Training phase
				while True:
					try:
						[filename, image, findings, bboxes] = sess.run(next_element_train)
						
						# print(findings, bboxes)
						# print(findings.shape, bboxes.shape)
						# exit()

						findings_ = []
						for finding in findings:
							findings_.append(dataset.label_to_name(label_df,finding[0,...]))

						visualize.draw_boxes(image[0,...],boxes=bboxes[0,...],captions=findings_)

						# print(findings)
						# print(bboxes)

						# findings_.append("test")
						# bboxes = np.append(bboxes,[[[0,0,256,512]]],axis=1)
						# visualize.draw_boxes(image[0,...],boxes=bboxes[0,...],captions=findings_, visibilities=[1,3])
					
					except tf.errors.OutOfRangeError:
						break

			
			return