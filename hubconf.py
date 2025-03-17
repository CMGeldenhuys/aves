#!/usr/bin/env python3
dependencies = ["torch", "torchaudio"]

import os
import sys
import tempfile
import json
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple

from torch import Tensor
import torch.hub as torchhub
from torch.hub import download_url_to_file, load_state_dict_from_url
from torch.nn import Module
from torchaudio.models import Wav2Vec2Model, wav2vec2_model

_AVES_CORE = "aves-base-core"
_AVES_BIO = "aves-base-bio"
_AVES_NONBIO = "aves-base-nonbio"
_AVES_ALL = "aves-base-all"


def _config_url(model):
    return f"https://storage.googleapis.com/esp-public-files/ported_aves/{model}.torchaudio.model_config.json"


def _model_url(model):
    return f"https://storage.googleapis.com/esp-public-files/ported_aves/{model}.torchaudio.pt"


_AVES_URLS = {
    _AVES_CORE: (_config_url(_AVES_CORE), _model_url(_AVES_CORE)),
    _AVES_BIO: (_config_url(_AVES_BIO), _model_url(_AVES_BIO)),
    _AVES_NONBIO: (_config_url(_AVES_NONBIO), _model_url(_AVES_NONBIO)),
    _AVES_ALL: (_config_url(_AVES_ALL), _model_url(_AVES_ALL)),
}


# Heavily based on https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/hub.py#L803
# Infact, not sure why this isn't already a utility function?..
def _fetch_aux_file_from_url(
    url: str,
    model_dir: Optional[str] = None,
    progress: bool = True,
    file_name: Optional[str] = None,
    check_hash: bool = False,
):
    if model_dir is None:
        hub_dir = torchhub.get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = torchhub.HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


# From README
def _load_config(config_path):
    with open(config_path, "r") as ff:
        obj = json.load(ff)

    return obj


class AvesModel(Module):
    def __init__(self, config: Dict, model: Wav2Vec2Model):
        super().__init__()

        # TODO: annotate types
        self.config = config
        self.model = model
        model.extract_features

    @classmethod
    def from_hub(cls, model_name, *, progress=True, **kwargs):
        config_url, model_url = _AVES_URLS[model_name]

        config_file = _fetch_aux_file_from_url(config_url, progress=progress)
        config = _load_config(config_file)

        state_dict = load_state_dict_from_url(model_url, progress=progress)

        inner_model = wav2vec2_model(**config, aux_num_out=None)
        inner_model.load_state_dict(state_dict)

        aves_model = cls(config, inner_model, **kwargs)
        return aves_model

    def extract_features(self, X: Tensor) -> Tuple[List[Tensor], Tensor | None]:
        return self.model.extract_features(X)

    def forward(self, X: Tensor) -> Tensor:
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        return self.extract_features(X)[0][-1]
