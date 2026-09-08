# Copyright (c) OpenMMLab. All rights reserved.
"""Small compatibility helper around ``torch.load``.

PyTorch only added the ``weights_only`` keyword argument to ``torch.load`` in
1.13 (and later changed its default). This repo is still validated against
PyTorch 1.11 (see ``tools/setup_codetr_env.sh``), so any call site that wants
to be explicit about ``weights_only`` needs to first check whether the
installed PyTorch actually supports it, otherwise the call raises
``TypeError: 'weights_only' is an invalid keyword argument``.
"""
import inspect

import torch

_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = 'weights_only' in inspect.signature(
    torch.load).parameters


def torch_load_compat(*args, weights_only=False, **kwargs):
    """Wrapper around ``torch.load`` that only forwards ``weights_only`` when
    the installed PyTorch version supports it.

    Checkpoints handled by this repo (mmdet/mmcv checkpoints, third-party
    converter inputs) generally contain more than plain tensors (e.g. a
    ``meta`` dict with config strings), so the default here mirrors the
    legacy ``torch.load`` behavior (``weights_only=False``) rather than
    PyTorch's newer, stricter default.
    """
    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        kwargs['weights_only'] = weights_only
    return torch.load(*args, **kwargs)
