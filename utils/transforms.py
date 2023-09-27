from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import (
    Transform,
    MapTransform,
    RandCoarseDropout,
    RandCoarseShuffle,
    RandomizableTransform,
)
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor

from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import PreprocManager
from torch_ecg.augmenters import AugmenterManager

class TorchECGPreprocess(Transform):
    def __init__(self, sampling_frequency=500, **kwargs) -> None:
        self.sampling_frequency = sampling_frequency
        
        cfg = CFG(**kwargs)
        ppm = PreprocManager.from_config(cfg)
        self.preprocessor = ppm
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        
        img, _ = self.preprocessor(img, self.sampling_frequency)
        
        out = convert_to_dst_type(img, img, dtype=img.dtype)[0]
        return out
    
class TorchECGPreprocessd(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = TorchECGPreprocess(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d 
    
    
class TorchECGAugment(Transform):
    def __init__(self, **kwargs) -> None:
        
        cfg = CFG(**kwargs)
        aug = AugmenterManager.from_config(cfg)
        self.augmenter = aug
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        
        img, _ = self.augmenter(img, None)
        
        out = convert_to_dst_type(img, img, dtype=img.dtype)[0]
        return out
    
class TorchECGAugmentd(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = TorchECGAugment(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d 
    
    
class RandCoarseDropoutChannelwise(RandCoarseDropout):
    """
    MONAI's RandCoarseDropout + Channelwise holes

    """

    def __init__(
        self,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
    ) -> None:
        super().__init__(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=prob
        )
        self.dropout_holes = dropout_holes
        if isinstance(fill_value, (tuple, list)):
            if len(fill_value) != 2:
                raise ValueError("fill value should contain 2 numbers if providing the `min` and `max`.")
        self.fill_value = fill_value
        self.prob_channelwise = prob_channelwise

    def _transform_holes(self, img: np.ndarray):
        """
        Fill the randomly selected `self.hole_coords` in input images.
        Please note that we usually only use `self.R` in `randomize()` method, here is a special case.

        """
        fill_value = (img.min(), img.max()) if self.fill_value is None else self.fill_value

        if self.dropout_holes:
            for h in self.hole_coords:
                for i, _ in enumerate(img[h]):
                    if self.R.uniform() < self.prob_channelwise:
                        if isinstance(fill_value, (tuple, list)):
                            img[h][i] = self.R.uniform(fill_value[0], fill_value[1], size=img[h][i].shape)
                        else:
                            img[h][i] = fill_value
            ret = img
        else:
            if isinstance(fill_value, (tuple, list)):
                ret = self.R.uniform(fill_value[0], fill_value[1], size=img.shape).astype(img.dtype, copy=False)
            else:
                ret = np.full_like(img, fill_value)
            for h in self.hole_coords:
                for i, _ in enumerate(img[h]):
                    if self.R.uniform() < self.prob_channelwise:
                        ret[h][i] = img[h][i]
        return ret
    
class RandCoarseDropoutChannelwised(RandomizableTransform, MapTransform):

    backend = RandCoarseDropoutChannelwise.backend

    def __init__(
        self,
        keys: KeysCollection,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.dropper = RandCoarseDropoutChannelwise(
            holes=holes,
            spatial_size=spatial_size,
            dropout_holes=dropout_holes,
            fill_value=fill_value,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
            prob=1.0,
            prob_channelwise=prob_channelwise,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCoarseDropoutChannelwised:
        super().set_random_state(seed, state)
        self.dropper.set_random_state(seed, state)
        return self


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.dropper.randomize(d[first_key].shape[1:])
        for key in self.key_iterator(d):
            d[key] = self.dropper(img=d[key], randomize=False)

        return d
    
class RandCoarseShuffleChannelwise(RandCoarseShuffle):
    def __init__(
        self,
        holes: int,
        spatial_size: Sequence[int] | int,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
    ) -> None:
        super().__init__(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=prob
        )
        self.prob_channelwise = prob_channelwise

    def _transform_holes(self, img: np.ndarray):
        """
        Shuffle the content of randomly selected `self.hole_coords` in input images.
        Please note that we usually only use `self.R` in `randomize()` method, here is a special case.

        """
        for h in self.hole_coords:
            # shuffle every channel separately
            for i, c in enumerate(img[h]):
                if self.R.uniform() < self.prob_channelwise:
                    patch_channel = c.flatten()
                    self.R.shuffle(patch_channel)
                    img[h][i] = patch_channel.reshape(c.shape)
        return img
    
class RandCoarseShuffleChannelwised(RandomizableTransform, MapTransform):

    backend = RandCoarseShuffleChannelwise.backend

    def __init__(
        self,
        keys: KeysCollection,
        holes: int,
        spatial_size: Sequence[int] | int,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.shuffle = RandCoarseShuffleChannelwise(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=1.0, prob_channelwise=prob_channelwise,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCoarseShuffleChannelwised:
        super().set_random_state(seed, state)
        self.shuffle.set_random_state(seed, state)
        return self


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.shuffle.randomize(d[first_key].shape[1:])
        for key in self.key_iterator(d):
            d[key] = self.shuffle(img=d[key], randomize=False)

        return d