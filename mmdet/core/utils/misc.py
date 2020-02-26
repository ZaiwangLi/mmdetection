# transpose + contiguous
# *args: no default values
# **kwargs: key value pair with defults
# map: 
# >>>def square(x) :            # 计算平方数
# ...     return x ** 2
# ... 
# >>> map(square, [1,2,3,4,5])   # 计算列表各个元素的平方
# [1, 4, 9, 16, 25]


from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


# an example of tranpose + contiguous
def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    # **kwargs are the func parameters, *args are multiple intputs
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        # all-zero tensor
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
