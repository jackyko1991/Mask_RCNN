import argparse
import train
import os
import dataset
import tensorflow as tf
import Config

def parser():
	parser = argparse.ArgumentParser(
		prog="NIH_chest_trainer",
		formatter_class=argparse.MetavarTypeHelpFormatter,
		description="Mask RCNN training on NIH chest xray dataset")
	parser.add_argument('--phase', default="train",
		dest="phase", type=str, help="learning phase (train or evaluate)")
	parser.add_argument('--train-data-dir', default="./data/train_mini",
		dest="train_data_dir", type=str, help="folder containing training dataset")
	parser.add_argument('--test-data-dir', default="./data/test",
		dest="test_data_dir", type=str, help="folder containing testing dataset")
	parser.add_argument('--eval-data-dir', default="./data/eval",
		dest="eval_data_dir", type=str, help="folder containing evaluation dataset")
	parser.add_argument('--bbox-csv', default="D:/projects/NIH_chest_xray/BBox_List_2017.csv",
		dest="bbox_csv", type=str, help="bbox csv file")
	parser.add_argument('--data-entry-csv', default="D:/projects/NIH_chest_xray/Data_Entry_2017.csv",
		dest="data_entry_csv", type=str, help="data entry csv file")
	parser.add_argument('--label-csv', default="./label.csv",
		dest="label_csv", type=str, help="label csv file")
	parser.add_argument('--config', default="./config.json",
		dest="config", type=str, help="config file location")
	parser.add_argument('--format', default="png",
		dest="format", type=str, help="chest xray image format (png/dcm)")
	args = parser.parse_args()
	return args

def main():
	args = parser()
	with tf.Graph().as_default():
		if (args.phase == "train"):
			print("Start training...")
			trainer = train.Trainer()

			# set config file
			config = Config.JsonConfig()
			config.readJson(args.config)
			trainer.config = config
			
			# dataset pipeline
			# Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
			with tf.device('/cpu:0'):
				# create transformations to image and labels
				trainTransforms = [
					dataset.BboxNihToMrcnn()
					# NiftiDataset.StatisticalNormalization(2.5),
					# # NiftiDataset.Normalization(),
					# NiftiDataset.Resample((0.45,0.45,0.45)),
					# NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
					# NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
					# NiftiDataset.RandomNoise()
					]

				TrainDataset = dataset.Dataset(
					data_dir=args.train_data_dir,
					transforms=trainTransforms,
				    train=True,
				    image_format="png",
				    bbox_csv=args.bbox_csv,
				    data_entry_csv=args.data_entry_csv,
				    label_csv=args.label_csv,
				    no_finding_prob=0.0
					)

				trainDataset = TrainDataset.get_dataset()
				# trainDataset = trainDataset.shuffle(buffer_size=5)
				trainDataset = trainDataset.batch(config.getConfigValue("BATCH_SIZE"))
				trainer.trainDataset = trainDataset

			# run training
			trainer.train()

		elif (args.phase == "evaluate"):
			print("Start evaluation...")
			return
		else:
			print("Training phase should be train or evaluate, abort")
			return

if __name__=="__main__":
	main()