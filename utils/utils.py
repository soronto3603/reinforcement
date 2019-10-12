from os import listdir
from os.path import isfile, join

def get_last_weights_filepath(path):
  onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
  return path + sorted(onlyfiles, reverse=True)[0]
