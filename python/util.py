import multiprocessing as mp
import numpy as np

def get_steps(arr):
    return tuple(np.append(np.cumprod(np.array(arr.shape)[1:][::-1])[::-1], 1))

def get_image_idxs(im, max_idx, min_idx=(0,0,0)):
    xs, ys, zs = np.ix_(range(min_idx[0], max_idx[0] + 1), range(min_idx[1], max_idx[1] + 1),
                    range(min_idx[2], max_idx[2] + 1))
    steps = get_steps(im)
    return np.array(np.unravel_index((xs * steps[0] + ys * steps[1] + zs * steps[2]).flatten(), im.shape))

def get_target_affinities(seg, idxs):
    aff = np.empty((len(idxs[0]), 3), dtype=bool)
    aff[:, 0] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[1], [0], [0]])])
    aff[:, 1] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[0], [1], [0]])])
    aff[:, 2] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[0], [0], [1]])])
    return aff

def entropy(lst):
    if lst.size == 0:
        return 0
    else:
        b1 = np.bincount(lst[:, 0]) / float(lst[:, 0].size)
        b2 = np.bincount(lst[:, 1]) / float(lst[:, 1].size)
        b3 = np.bincount(lst[:, 2]) / float(lst[:, 2].size)
        vfunc = np.vectorize(lambda x: 0 if x == 0 else -x * np.log(x))
        return np.sum(vfunc(b1)) + np.sum(vfunc(b2)) + np.sum(vfunc(b3))

def proportion(lst):
    if lst.size == 0:
        return [0, 0, 0]
    else:
        p1 = np.count_nonzero(lst[:, 0]) / float(lst[:, 0].size)
        p2 = np.count_nonzero(lst[:, 1]) / float(lst[:, 1].size)
        p3 = np.count_nonzero(lst[:, 2]) / float(lst[:, 2].size)
    return [p1, p2, p3]


# Parallel shit
def chunk(lst, num_chunks):
    chunks = [lst[i*len(lst)/num_chunks:(i+1)*len(lst)/num_chunks] for i in range(0, num_chunks)]
    return filter(lambda x: x != [], chunks)

def par_max_by(arr, pool_size, func, func_args, max_key):
    chunks = chunk(arr, pool_size)
    dom = [(chunks[i], func, func_args, max_key) for i in range(len(chunks))]
    mapped = maybe_par_map(_par_max_inner, dom, pool_size)
    return max(mapped, key=max_key)

def par_sum(arr, pool_size, func, func_args):
    chunks = chunk(arr, pool_size)
    dom = [(chunks[i], func, func_args) for i in range(len(chunks))]
    mapped = maybe_par_map(_par_sum_inner, dom, pool_size)
    return sum(mapped)

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
    first = func(arr[0], *func_args)
    return reduce(lambda x, y: max([x, func(y, *func_args)], key = max_key), arr[1:], first)

def _par_sum_inner(args):
    arr, func, func_args = args
    first = func(arr[0], *func_args)
    return reduce(lambda x, y: sum([x, func(y, *func_args)]), arr[1:], first)