import os
import Config
import datetime
import tensorflow as tf

class Trainer:
	def __init__(self):
		self.config = None
		self.trainDataset = None
		self.testDataset = None

	def train(self):
		# load data
		train_iterator = self.trainDataset.make_initializable_iterator()
		next_element_train = train_iterator.get_next()

		# training cycle
		with tf.Session() as sess:
			# Initialize all variables
			sess.run(tf.global_variables_initializer())
			print("{}: Start training...".format(datetime.datetime.now()))

			sess.run(train_iterator.initializer)
			[image, finding, bbox] = sess.run(next_element_train)
			print(image.shape, finding.shape, bbox.shape)
			return

