# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
from typing import Sequence, Union, Optional, List, Tuple

from mmengine.dataset import BaseDataset, force_full_init

#from mmdet.registry import DATASETS, TRANSFORMS
from mmdet.registry import TRANSFORMS
import numpy as np
import math
from mmdet.datasets import ConcatDataset
from mmdet.datasets import CocoDataset
import random
import os
from mmyolo.registry import DATASETS


@DATASETS.register_module()
class NODDataset(CocoDataset):
    """Dataset for COCO."""
    """For Sony"""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(0, 0, 142), (220, 20, 60),(106, 0, 228),]
    }

    def __init__(self,
                data_root: str,
                ann_file: str,
                dataset_name: Optional[str] = None,
                normalization_align_type=None,
                upsample_factor=None,
                replace_filename=None,
                **kwargs) -> None:

        if dataset_name is not None:
            data_root = os.path.join(data_root, dataset_name)

        self.replace_filename = replace_filename
        self.normalization_align_type = normalization_align_type
        self.upsample_factor = upsample_factor

        self.normalization_data = None

        if self.normalization_align_type is not None:
            if isinstance(self.normalization_align_type, tuple):
                if isinstance(self.normalization_align_type[0], tuple):
                    (min_source, max_source), (min_target, max_target) = self.normalization_align_type
                scale = (max_target - min_target) / (max_source - min_source)
                offset = min_target - min_source * scale
                normalization_data = (scale, offset)
            else:
                annotation_name = os.path.splitext(os.path.basename(ann_file))[0]
                normalization_path = os.path.join(data_root, "normalization", f"{annotation_name}_align_{self.normalization_align_type}.npy")
                normalization_data = np.load(normalization_path)
            self.normalization_data = normalization_data # (scale, offset)

        print(f"normalization_data: {self.normalization_data}")

        super().__init__(data_root=data_root, ann_file=ann_file, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:

        raw_data_info = super().parse_data_info(raw_data_info)
        

        img_path = raw_data_info['img_path']

        if self.replace_filename:
            for k, v in self.replace_filename.items():
                img_path = img_path.replace(k, v)
        raw_data_info['img_path'] = img_path

        if self.normalization_data is not None:
            raw_data_info["normalize_reverse"] = self.normalization_data

        if self.upsample_factor is not None:
            raw_data_info["upsample_factor"] = self.upsample_factor

        return raw_data_info
        
    def filter_data(self) -> List[dict]:
        data_list = super().filter_data()
    
        valid_data_infos = []
        for i, data_info in enumerate(data_list):
            img_path = data_info['img_path']

            if not os.path.exists(img_path):
                continue

            valid_data_infos.append(data_info)

        print(f"filtered {len(data_list)} to {len(valid_data_infos)}")

        return valid_data_infos


@DATASETS.register_module()
class NODDataset_Nikon(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(0, 0, 142), (220, 20, 60),(106, 0, 228),]
    }

    def __init__(self,
                data_root: str,
                ann_file: str,
                dataset_name: Optional[str] = None,
                normalization_align_type=None,
                replace_filename=None,
                **kwargs) -> None:

        if dataset_name is not None:
            data_root = os.path.join(data_root, dataset_name)

        self.replace_filename = replace_filename
        self.normalization_align_type = normalization_align_type

        self.normalization_data = None

        if self.normalization_align_type is not None:
            annotation_name = os.path.splitext(os.path.basename(ann_file))[0]
            normalization_path = os.path.join(data_root, "normalization", f"train_align_{self.normalization_align_type}.npy")
            normalization_data = np.load(normalization_path)
            self.normalization_data = normalization_data # (scale, offset)

        print(f"normalization_data: {self.normalization_data}")

        super().__init__(data_root=data_root, ann_file=ann_file, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:

        raw_data_info = super().parse_data_info(raw_data_info)
        

        img_path = raw_data_info['img_path']

        if self.replace_filename:
            for k, v in self.replace_filename.items():
                img_path = img_path.replace(k, v)
        raw_data_info['img_path'] = img_path

        if self.normalization_data is not None:
            raw_data_info["normalize_reverse"] = self.normalization_data

        return raw_data_info
        
    def filter_data(self) -> List[dict]:
        data_list = super().filter_data()
    
        valid_data_infos = []
        for i, data_info in enumerate(data_list):
            img_path = data_info['img_path']

            if not os.path.exists(img_path):
                continue

            valid_data_infos.append(data_info)

        print(f"filtered {len(data_list)} to {len(valid_data_infos)}")

        return valid_data_infos

@DATASETS.register_module()
class LimitedDataset:

    def __init__(
            self,
            dataset: Union[BaseDataset, dict],
            max_items: Optional[int] = None,
            seed: int = 0,
            target_size: Optional[float] = None,
            keep_original_size = True
        ) -> None:

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.max_items = max_items

        if self.max_items is None:
            self._indices = np.arange(self.num_samples)

        else:

            num_samples_target = self.num_samples if target_size is None else int(self.num_samples * target_size)
            if num_samples_target < self.max_items:
                raise ValueError(
                    f"max_items {self.max_items} is larger than the number of samples {num_samples_target} in the dataset.")
            
            rdn = np.random.RandomState(seed)
            _indices = rdn.choice(self.num_samples, self.max_items, replace=False)

            print(f"selected indices: {_indices}")

            if keep_original_size:
                repeat_factor = math.ceil(num_samples_target / self.max_items)
                _indices = np.tile(_indices, repeat_factor)[:num_samples_target]
                print(f"num repeated indices: {len(_indices)}")
            else:
                assert target_size is None

            self._indices = _indices

        for i in range(10):
            data = self.dataset[self._indices[i]]
            data_samples = data["data_samples"]
            # print(data)
            print(data_samples.img_id)

        self._fully_initialized = False
        self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self.dataset[self._indices[idx]]




@DATASETS.register_module()
class RandomSampleDataset:

    def __init__(
            self,
            dataset: Union[BaseDataset, dict],
            num_items: Optional[int] = None,
        ) -> None:

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.num_items = num_items

        self._fully_initialized = False
        self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return self.dataset[np.random.randint(0, self.num_samples)]
        # return self.dataset[self._indices[idx]]

@DATASETS.register_module()
class SampleConcatDataset(ConcatDataset):

    def __init__(self,
                n: int,
                f: Union[float, List[float]],
                datasets: Sequence[Union[BaseDataset, dict]],
                lazy_init: bool = False,
                ignore_keys: Union[str, List[str], None] = None
                ):

        super().__init__(datasets=datasets, lazy_init=lazy_init, ignore_keys=ignore_keys)

        if isinstance(f, float):
            f = [f, 1-f]

        assert len(f) == len(datasets)
        self.n = n
        self.f = f

        print(f"SampleConcatDataset: n={n}, f={f}")

    def __len__(self):
        return self.n

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> Tuple[int, int]:

        dataset_idx = random.choices(range(len(self.datasets)), weights=self.f)[0]
        dataset = self.datasets[dataset_idx]
        sample_idx = np.random.randint(0, len(dataset))

        return dataset_idx, sample_idx

