import numpy as np


def transform_depth(depth, params):
    # depth = mm_to_m(depth)
    if params["max_depth"]:
        depth = normalize_max_depth(depth, params["max_depth"])
    return depth

def normalize_max_depth(depth, max_depth):
    depth[depth > max_depth] = max_depth
    return depth


