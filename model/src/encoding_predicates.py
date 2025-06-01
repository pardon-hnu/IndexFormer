import re

import numpy as np



def get_representation(value, word_vectors):
    if value in word_vectors:
        embedded_result = np.array(list(word_vectors[value]))
    else:
        embedded_result = np.array([0.0 for _ in range(500)])
    hash_result = np.array([0.0 for _ in range(500)])
    for t in value:
        hash_result[hash(t) % 500] = 1.0
    return np.concatenate((embedded_result, hash_result), 0)


def get_str_representation(value, word_vectors):
    vec = np.array([])
    count = 0
    for v in value.split('%'):
        if len(v) > 0:
            if len(vec) == 0:
                vec = get_representation(v, word_vectors)
                count = 1
            else:
                new_vec = get_representation(v, word_vectors)
                vec = vec + new_vec
                count += 1
    if count > 0:
        vec /= float(count)
    return vec


