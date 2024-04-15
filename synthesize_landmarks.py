#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

from misc import compute_mirror_error

li = [17286,17577,17765,17885,18012,18542,18668,18788,18987,19236,7882,7896,7905,7911,6479,7323,
      7922,8523,9362,1586,3480,4770,5807,4266,3236, 10176,11203,12364,14269,12636,11602,5243,5875,
      7096,7936,9016,10244,10644,9638,8796,7956,7116,6269,5629,6985,7945,8905,10386,8669,7949,7229]

li = np.array(li)

parser = argparse.ArgumentParser()
parser.add_argument('--facial_features', type=str, default='lb,rb,re,le,no,ul,ll')
parser.add_argument('--symmetry_alpha', type=float, default=0, 
                    help="""A coefficient that takes values between 0 and 1. When set to 0, symmetry is not enforced.
                    When set to 1, complete symmetry is enforced.""")

args = parser.parse_args()

alpha = args.symmetry_alpha

rel_ids   = {'lb': np.array(list(range(0, 5))),
             'rb': np.array(list(range(5, 10))),
             'no': np.array(list(range(10, 19))),
             'le': np.array(list(range(19, 25))),
             're': np.array(list(range(25, 31))),
             'ul': np.array(list(range(31, 37))+list(range(43, 47))),
             'll': np.array(list(range(37, 43))+list(range(47, 51)))}

facial_features = args.facial_features.split(',')

morphable_model ='BFMmm-19830' # sys.argv[2] # 'BFMmm-19830'
localized_basis_file = f'models/MMs/{morphable_model}/E/localized_basis/v.0.0.1.12-defaultd.npy'

basis_set = np.load(localized_basis_file, allow_pickle=True).item()
sdir = f'models/MMs/{morphable_model}/'

P0 = np.loadtxt(f'{sdir}/p0L_mat.dat')
X0 = P0[:,0]
Y0 = P0[:,1]
Z0 = P0[:,2]

EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li,:]
EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li,:]
EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li,:]

L0 = np.concatenate((X0.reshape(-1,1), Y0.reshape(-1,1), Z0.reshape(-1,1)), axis=1)
L0 = L0.reshape(-1,1)
Efull = np.concatenate((EX, EY, EZ), axis=0)

bix = 0
p0 = np.loadtxt(f'./models/MMs/{morphable_model}/p0L_mat.dat')
deltaL = np.zeros((1, 51*3))

for feat in facial_features:
    rel_id = rel_ids[feat]
    
    for k in range(0, basis_set['num_comps'][feat]):
        xmin = basis_set[feat].stats['min_0.5pctl'][k]
        xmax = basis_set[feat].stats['max_99.5pctl'][k]
        xmean = basis_set[feat].stats['mean'][k]
        xstd = basis_set[feat].stats['std'][k]
        
        xdiff = (xmax-xmin)/2.0
        xmean = (xmax+xmin)/2.0
        
        
        if feat in ['ll','ul']:
            nsigmas = 1.5
        elif feat in ['lb','rb']:
            nsigmas = 4
        else:
            nsigmas = 3
            
        x = nsigmas*xstd*np.random.randn()+xmean
        
        de = (1+alpha)*x*basis_set[feat].components_[k:(k+1),:]
        
        deltaL[:,rel_id*3] += de[:,0:int(de.shape[1]/3)]
        deltaL[:,1+rel_id*3] += de[:,int(de.shape[1]/3):int(de.shape[1]/3)*2]
        deltaL[:,2+rel_id*3] += de[:,int(de.shape[1]/3)*2:]

deltaL = deltaL.reshape(-1,1)
L0 = L0.reshape(-1,3)
deltaL = deltaL.reshape(-1,3)

L = L0+deltaL

rel_ids_rev = {
    'lb': np.array([9,8,7,6,5]),
    'le': np.array([28, 27, 26, 25, 30, 29]),
    'no': np.array([10, 11, 12, 13, 18, 17, 16, 15, 14]),
    'll': np.array([31, 42, 41, 40, 39, 38, 43, 50, 49, 48]),
    'ul': np.array([37, 36, 35, 34, 33, 32, 47, 46, 45, 44]),
    'rb': np.array([4,3,2,1,0]),
    're': np.array([22, 21, 20, 19, 24, 23]),
    }

mirror_mx = np.eye(3)
mirror_mx[0,0] = -1

L0_ = copy.deepcopy(L0)

"""
for feat in facial_features:
    mag = np.linalg.norm(deltaL[rel_ids[feat],:])
    print(f'movement magnitude for feature {feat}: {mag} mm (all landmarks)' ) 
"""

norm_coeff = (1+alpha)

L0_[rel_ids['lb'],:] = (L0[rel_ids['lb'],:] + L0[rel_ids_rev['lb'],:]@mirror_mx)
L0_[rel_ids['rb'],:] = (L0[rel_ids['rb'],:] + L0[rel_ids_rev['rb'],:]@mirror_mx) 
L0_[rel_ids['le'],:] = (L0[rel_ids['le'],:] + L0[rel_ids_rev['le'],:]@mirror_mx) 
L0_[rel_ids['re'],:] = (L0[rel_ids['re'],:] + L0[rel_ids_rev['re'],:]@mirror_mx)
L0_[rel_ids['no'],:] = (L0[rel_ids['no'],:] + L0[rel_ids_rev['no'],:]@mirror_mx)
L0_[rel_ids['ll'],:] = (L0[rel_ids['ll'],:] + L0[rel_ids_rev['ll'],:]@mirror_mx)
L0_[rel_ids['ul'],:] = (L0[rel_ids['ul'],:] + L0[rel_ids_rev['ul'],:]@mirror_mx)

L_ = copy.deepcopy(L0_)

L_[rel_ids['lb'],:] += (deltaL[rel_ids['lb'],:] + alpha*deltaL[rel_ids_rev['lb'],:]@mirror_mx)/norm_coeff
L_[rel_ids['rb'],:] += (deltaL[rel_ids['rb'],:] + alpha*deltaL[rel_ids_rev['rb'],:]@mirror_mx)/norm_coeff
L_[rel_ids['le'],:] += (deltaL[rel_ids['le'],:] + alpha*deltaL[rel_ids_rev['le'],:]@mirror_mx)/norm_coeff
L_[rel_ids['re'],:] += (deltaL[rel_ids['re'],:] + alpha*deltaL[rel_ids_rev['re'],:]@mirror_mx)/norm_coeff
L_[rel_ids['no'],:] += (deltaL[rel_ids['no'],:] + alpha*deltaL[rel_ids_rev['no'],:]@mirror_mx)/norm_coeff
L_[rel_ids['ll'],:] += (deltaL[rel_ids['ll'],:] + alpha*deltaL[rel_ids_rev['ll'],:]@mirror_mx)/norm_coeff
L_[rel_ids['ul'],:] += (deltaL[rel_ids['ul'],:] + alpha*deltaL[rel_ids_rev['ul'],:]@mirror_mx)/norm_coeff


mirror_error = compute_mirror_error(L_)
print("Average mirror errors (in mms):")
print(mirror_error)

plt.figure(figsize=(7,7))
plt.plot(L0_[:,0], -L0_[:,1], 'x')
plt.plot(L_[:,0], -L_[:,1], 'o')
plt.show()
