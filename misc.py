"""
Created on Mon Apr 15 11:36:20 2024

@author: sariyanide
"""

import numpy as np

def compute_mirror_error(L):
    
    ul = np.array(list(range(31, 37))+list(range(43, 47)))
    ll = np.array(list(range(37, 43))+list(range(47, 51)))
    
    rel_ids   = {'lb': np.array(list(range(0, 5))),
                 'rb': np.array(list(range(5, 10))),
                 'no': np.array(list(range(10, 19))),
                 'le': np.array(list(range(19, 25))),
                 're': np.array(list(range(25, 31))),
                 'mo': np.concatenate((ul, ll))}
    
    ll_rev = np.array([31, 42, 41, 40, 39, 38, 43, 50, 49, 48])
    ul_rev = np.array([37, 36, 35, 34, 33, 32, 47, 46, 45, 44])
    
    rel_ids_rev = {
        'lb': np.array([9,8,7,6,5]),
        'le': np.array([28, 27, 26, 25, 30, 29]),
        'no': np.array([10, 11, 12, 13, 18, 17, 16, 15, 14]),
        'rb': np.array([4,3,2,1,0]),
        're': np.array([22, 21, 20, 19, 24, 23]),
        'mo': np.concatenate((ul_rev, ll_rev))}

    mirror_mx = np.eye(3)
    mirror_mx[0,0] = -1
    
    Lmirrored = np.zeros(L.shape)
    
    for feat in rel_ids:
        Lmirrored[rel_ids[feat],:] = L[rel_ids_rev[feat],:]@mirror_mx
    
    # import matplotlib.pyplot as plt
    # plt.plot(L[:,0], -L[:,1], 'x')
    # plt.plot(Lmirrored[:,0], -Lmirrored[:,1], 'x')
    # plt.show()
    
    mirror_error = {'total': 0}
    
    # Compute mirrored error
    for feat in rel_ids:
        cur = np.mean(np.sqrt(np.sum((Lmirrored[rel_ids[feat],:]-L[rel_ids[feat],:])**2,axis=1)))
        mirror_error[feat] = cur
        mirror_error['total'] += cur
    
    return mirror_error         

    
    
