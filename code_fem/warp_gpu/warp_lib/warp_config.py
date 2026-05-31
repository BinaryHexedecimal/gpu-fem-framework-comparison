
import warp as wp

USE_FP64 = True

scalar = wp.float64
vec3   = wp.vec3d
mat33  = wp.mat33d


def update_types():
    global scalar, vec3, mat33
    scalar = wp.float64 if USE_FP64 else wp.float32
    vec3   = wp.vec3d   if USE_FP64 else wp.vec3
    mat33  = wp.mat33d  if USE_FP64 else wp.mat33