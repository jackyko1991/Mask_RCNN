import argparse
import tarfile
import os
import shutil
import io
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

def parser():
	parser = argparse.ArgumentParser(
		prog="NIH_chest_data_prepare",
		formatter_class=argparse.MetavarTypeHelpFormatter,
		description="Prepare NIH chest xray data")
	parser.add_argument('--phase', default="train",
		dest="phase", type=str, help="learning phase (train or evaluate)")
	# parser.add_argument('--raw-data-dir', default="./raw_data",
	# 	dest="raw_data_dir", type=str, help="folder containing raw dataset")
	parser.add_argument('--raw-data-dir', default="D:/projects/NIH_chest_xray",
		dest="raw_data_dir", type=str, help="folder containing raw dataset")
	parser.add_argument('--train-data-dir', default="./data/train",
		dest="train_data_dir", type=str, help="folder containing training dataset")
	parser.add_argument('--test-data-dir', default="./data/test",
		dest="test_data_dir", type=str, help="folder containing testing dataset")
	parser.add_argument('--eval-data-dir', default="./data/eval",
		dest="eval_data_dir", type=str, help="folder containing evaluation dataset")
	parser.add_argument('--train-test-split', default=0.2,
		dest="train_test_split", type=float, help=" proportion of the dataset to include in the test split, value should between 0 and 1")
	parser.add_argument('--cache', default="./tmp",
		dest="cache", type=str, help="uncompress cache folder")
	parser.add_argument('--format', default="png",
		dest="format", type=str, help="chest xray image format (png/dcm), only for GCP download")
	parser.add_argument('--gcp', default=False,
		dest="gcp", type=bool, help="use Google Cloud Storage to download data")
	args = parser.parse_args()
	return args

def get_file_progress_file_object_class(on_progress):
	class FileProgressFileObject(tarfile.ExFileObject):
		def read(self, size, *args):
			on_progress(self.name, self.position, self.size)
			return tarfile.ExFileObject.read(self, size, *args)
	return FileProgressFileObject

class TestFileProgressFileObject(tarfile.ExFileObject):
	def read(self, size, *args):
		on_progress(self.name, self.position, self.size)
		return tarfile.ExFileObject.read(self, size, *args)

class ProgressFileObject(io.FileIO):
	def __init__(self, path, *args, **kwargs):
		self._total_size = os.path.getsize(path)
		io.FileIO.__init__(self, path, *args, **kwargs)

	def read(self, size):
		print("Overall process: %d of %d" %(self.tell(), self._total_size))
		return io.FileIO.read(self, size)

def on_progress(filename, position, total_size):
	print("%s: %d of %s" %(filename, position, total_size))

def src_dest_path_map(data_list, src_dir, dest_dir):
	path_map = []
	for data in data_list:
		path_map.append((os.path.join(src_dir, data),os.path.join(dest_dir, data)))

	return path_map

def process_local(args):
	print("Processing local data...")

	raw_data_dir = os.path.join(args.raw_data_dir,"images")

	# check necessary files exists
	# check for folder is empty
	if len(os.listdir(raw_data_dir)) == 0:
		print(raw_data_dir,"is empty, data processing abort")
		exit()

	# check dataset list
	if not os.path.exists(os.path.join(args.raw_data_dir, "test_list.txt")):
		print("test_list.txt not found, abort")
		exit()

	if not os.path.exists(os.path.join(args.raw_data_dir,"train_val_list.txt")):
		print("train_val_list.txt not found, abort")
		exit()
	
	# uncompress zipped imaged files
	tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)
	for file in os.listdir(os.path.join(args.raw_data_dir,"images")):

		print("Start uncompress",file,"...")
		tar = tarfile.open(fileobj=ProgressFileObject(os.path.join(raw_data_dir, file)))
		tar.extractall(path=args.cache)
		tar.close()

	# read train/test and evaluation dataset list
	with open(os.path.join(args.raw_data_dir, "train_val_list.txt"), 'r') as f: 
		train_test_list = f.read().splitlines()

	# split train/test dataset
	# not sure if full dataset is downloaded, create local train/test list first
	train_test_list_local = []
	for data in os.listdir(os.path.join(args.cache,"images")):
		if data in train_test_list:
			train_test_list_local.append(data)

	train_list, test_list = train_test_split(train_test_list_local, test_size=args.train_test_split)

	with open(os.path.join(args.raw_data_dir, "test_list.txt"), 'r') as f: 
		eval_list = f.read().splitlines()

	eval_list_local = []
	for data in os.listdir(os.path.join(args.cache,"images")):
		if data in eval_list:
			eval_list_local.append(data)

	eval_list = eval_list_local

	# move data to target locations
	# prepare to src/dest absolute path
	train_list_map = src_dest_path_map(train_list, os.path.join(args.cache, "images"), args.train_data_dir)
	test_list_map = src_dest_path_map(test_list, os.path.join(args.cache, "images"), args.test_data_dir)
	eval_list_map = src_dest_path_map(eval_list, os.path.join(args.cache, "images"), args.eval_data_dir)

	# create target files before copy
	if not os.path.exists(args.train_data_dir):
		os.makedirs(args.train_data_dir)
	if not os.path.exists(args.test_data_dir):
		os.makedirs(args.test_data_dir)
	if not os.path.exists(args.eval_data_dir):
		os.makedirs(args.eval_data_dir)

	pool = Pool()
	print("Moving train data...")
	pool.starmap(shutil.move,train_list_map)
	print("Moving test data...")
	pool.starmap(shutil.move,test_list_map)
	print("Moving evaluation data...")
	pool.starmap(shutil.move,eval_list_map)

	# print(train_list_map)

	shutil.rmtree(args.cache)

	print("Data preparation complete")

	return

def main():
	args = parser()

	if args.gcp:
		print("Download from Google Cloud Storage coming soon")
		return
	else:
		process_local(args)

if __name__=="__main__":
	main()