import argparse

def parser():
	parser = argparse.ArgumentParser(
		prog="NIH_chest_trainer",
		formatter_class=argparse.MetavarTypeHelpFormatter,
		description="Mask RCNN training on NIH chest xray dataset")
	parser.add_argument('--phase', default="train",
		dest="phase", type=str, help="learning phase (train or evaluate)")
	parser.add_argument('--train-data-dir', default="./data/train",
		dest="train_data_dir", type=str, help="folder containing training dataset")
	parser.add_argument('--test-data-dir', default="./data/test",
		dest="test_data_dir", type=str, help="folder containing testing dataset")
	parser.add_argument('--format', default="png",
		dest="format", type=str, help="chest xray image format (png/dcm)")
	args = parser.parse_args()
	return args

def main():
	args = parser()


if __name__=="__main__":
	main()