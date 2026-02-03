import numpy as np
import hashlib

def get_hash_index(s, d):

    hash_val = int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
    return hash_val % d

def name2features(name):

    d = 8192
    v = np.zeros(d)
    name = name.lower()
    
    prefix_max = 4
    for m in range(1, prefix_max+1):
        if m <= len(name):
            prefix = name[:m]
            v[get_hash_index(f"prefix_{prefix}", d)] = 1
            if m == 1:
                v[get_hash_index(f"first_{prefix}", d)] = 1


    suffix_max = 4
    for m in range(1, suffix_max+1):
        if m <= len(name):
            suffix = name[-m:]
            v[get_hash_index(f"suffix_{suffix}", d)] = 1
            if m == 1:
                v[get_hash_index(f"last_{suffix}", d)] = 1


    for n_gram in [2, 3, 4]:
        for i in range(len(name)-n_gram+1):
            gram = name[i:i+n_gram]
            v[get_hash_index(f"gram_{gram}", d)] = 1

    return v