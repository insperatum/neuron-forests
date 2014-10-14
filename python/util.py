import multiprocessing as mp
import numpy as np

def get_steps(arr):
    return tuple(np.append(np.cumprod(np.array(arr.shape)[1:][::-1])[::-1], 1))

def entropy(lst):
    if lst.size == 0:
        return 0
    else:
        p = np.bincount(lst) / float(lst.size)
        vfunc = np.vectorize(lambda x: 0 if x == 0 else -x * np.log(x))
        return np.sum(vfunc(p))

def proportion(lst):
    return 0 if lst.size==0 else float(np.count_nonzero(lst)) / lst.size

def chunk(lst, num_chunks):
    return [lst[i*len(lst)/num_chunks:(i+1)*len(lst)/num_chunks] for i in range(0, num_chunks)]

def par_max_by(arr, pool_size, func, func_args, max_key):
    chunks = chunk(arr, pool_size)
    dom = [(chunks[i], func, func_args, max_key) for i in range(pool_size)]
    mapped = maybe_par_map(_par_max_inner, dom, pool_size)
    return max(mapped, key=max_key)

def maybe_par_map(f, dom, pool_size):
    if pool_size == 1:
        mapped = map(f, dom)
    else:
        pool = mp.Pool(pool_size)
        mapped = pool.map(f, dom)
        pool.close()
        pool.join()
    return mapped

def _par_max_inner(args):
    arr, func, func_args, max_key = args
    best = None
    for x in arr:
        val = func(x, *func_args)
        if best is None or max_key(val) > max_key(best):
            best = val
    return best