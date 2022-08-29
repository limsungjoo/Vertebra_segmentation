import pickle
import gzip


def save_pickle(fileapth,obj):
	# save and compress.
	with gzip.open(fileapth, 'wb') as f:
		pickle.dump(obj, f)

def read_pickle(fileapth):
	with gzip.open(fileapth,"rb") as f:
		data = pickle.load(f)

	return data