"""Shim: provide missing torch.cuda.memory allocator symbols for torch 2.7.x"""
import torch.cuda.memory as _m
if not hasattr(_m, '_cuda_beginAllocateCurrentThreadToPool'):
    _m._cuda_beginAllocateCurrentThreadToPool = lambda *a, **kw: None
if not hasattr(_m, '_cuda_endAllocateToPool'):
    _m._cuda_endAllocateToPool = lambda *a, **kw: None
if not hasattr(_m, '_cuda_releasePool'):
    _m._cuda_releasePool = lambda *a, **kw: None
