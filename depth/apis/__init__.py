# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_depther, init_depther
from .test import multi_gpu_test, single_gpu_test
from .train import  set_random_seed, train_depther

__all__ = [
    'set_random_seed', 'train_depther', 'init_depther',
    'inference_depther', 'multi_gpu_test', 'single_gpu_test',
]
