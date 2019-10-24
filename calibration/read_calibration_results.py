import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--n", type=int)

args = parser.parse_args()

dirname = 'cali%d/'%args.n

ret = pickle.load(open(dirname + 'ret.pkl', 'rb'), encoding='latin1')
mtx = pickle.load(open(dirname + 'mtx.pkl', 'rb'), encoding='latin1')
dist = pickle.load(open(dirname + 'dist.pkl', 'rb'), encoding='latin1')
rvecs = pickle.load(open(dirname + 'rvecs.pkl', 'rb'), encoding='latin1')
tvecs = pickle.load(open(dirname + 'tvecs.pkl', 'rb'), encoding='latin1')

print('Reprojection Error :\n', ret)
print('Camera Matrix :\n', mtx)
print('Distortion Coefficient :\n', dist)
print('Rotation Vectors :\n', rvecs)
print('Translation Vectors :\n', tvecs)
