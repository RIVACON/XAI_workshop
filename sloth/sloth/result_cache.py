import hashlib
import numpy as np

_cache = {}

def has_result(key):
    global _cache
    return key in _cache

def get_result(key):
    global _cache
    return _cache[key]

def add_result(key, result):
    global _cache
    _cache[key] = result

def _create_hashkey_np(data: np.ndarray):
    return hashlib.sha1(data.tobytes()).hexdigest()

