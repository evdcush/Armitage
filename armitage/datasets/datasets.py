"""This module is used to implement and register the custom datasets.

If OpenMMLab series repositries have supported the target dataset, for example,
CocoDataset. You can simply use it by setting ``type=mmdet.CocoDataset`` in the
config file.

If you want to do some small modifications to the existing dataset,
you can inherit from it and override its methods:

Examples:
    >>> from mmdet.datasets import CocoDataset as MMDetCocoDataset
    >>>
    >>> class CocoDataset(MMDetCocoDataset):
    >>>     def load_data_list(self):
    >>>         ...

Read about the ``BaseDataset`` in MMEngine's docs:
https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html


---

Interface
=========

Assumes data is stored in the following path::

    data
    ├── annotations
    │   ├── train.json
    ├── train
    │   ├── xxx/xxx_0.jpg
    │   ├── xxx/xxx_1.jpg
    │   ├── ...






"""
import torch

from mmengine import load
from mmengine.dataset import BaseDataset

from armitage.registry import DATASETS

## SKIP THE BaseDataset IMPLEMENTATION FOR NOW!
'''
@DATASETS.register_module()
class BDD100K(BaseDataset):
    def __init__(self):
        pass
'''

@DATASETS.register_module()
class BDD100KDataset(torch.utils.data.Dataset):
    """BDD100K Dataset.

    This dataset has over 100k videos (each 40s long) recorded at 720p in 30hz.
    (Please see __file__.__doc__ for more details on the BDD100K Dataset.)

    .. note::
        Not all videos, nor their constituent frames, are used at the full 30hz
        for all labels.

    This dataset focuses on the **images** of BDD100K; it does not support
    reading of video files.

    This particular dataset assumes you are working with instance labels,
    specifically Det20, which has the file-structure::

        bdd100k/
        ├── images/
        │   └── 100k/
        │       ├── test/
        │       ├── train/
        │       └── val/
        └── labels/
            └── det_20/
                ├── det_train.json
                └── det_val.json

    """
    labels_10k = frozenset({
        "ins_seg",  # Instance Segmentation
        "pan_seg",  # Panoptic Segmentation
        "sem_seg",  # Semantic Segmentation
    })
    labels_100k = frozenset({
        "det_20",    # Object Detection
        "drivable",  # Drivable Area
        "lane",      # Lane Marking
        "pose_21",   # Pose Estimation
    })

    def __init__(self, images_path, labels_path):
        """
        Parameters
        ----------
        images_path : str
            Path to the root images directory.
        labels_path : str
            Path to the labels json.
        """
        super().__init__()
        self.annos_path = labels_path
        self.images_path = images_path

        labels = self.load_labels(labels_path)
        assert labels
        processed_labels = self.process_labels_det20(labels)
        assert processed_labels
        self._labels = processed_labels

    def load_labels(self, labels_path: str) -> list[dict]:
        """ Load labels json from given abspath. """
        #labels = dict()
        #with open(labels_path) as labels_json_file:
        #    labels = json.load(labels_json_file)
        #return labels
        return load(labels_path)

    def process_labels_det20(self, samples: list[dict]) -> list[dict]:
        """

        Parameters
        ==========
        samples : list[dict]
            The list of of samples loaded from the labels json; provided as a
            list of dictionaries, with the following items:
                name : str
                    File basename; eg 'b1c81faa-3df17267.jpg'
                attributes : dict
                    Scene/image-level attributes describing:
                    weather, timeofday, scene; eg:
                    {'weather': 'clear',
                     'timeofday': 'night',
                     'scene': 'highway'}
                timestamp : int
                    The timestamp in the respective video from which the image
                    frame was extracted; always ``10000`` for ``100k``-based
                    datasets.
                labels : list[dict]
                    Instance-level labels with the following items:
                        id : str (int)
                        attributes: dict
                            Instance-level attributes, describing:
                            occluded, truncated, trafficLightColor; eg:
                            {'occluded': False,
                             'truncated': False,
                             'trafficLightColor': 'NA'}
                        category : str
                            Object category; eg: 'car' or 'taffic light' etc..
                        box2d : dict[str, float]
                            Bounding box for object in xyxy format, keyed
                            literally as: 'x1', 'y1', 'x2', 'y2'

        Returns
        =======
        processed_samples : list[dict]
            The (lightly) processed set of samples. The key differences from an
            unprocessed sample are:
                - resolved file-path for the image
                - overall more flattened (eg dict[dict] --> dict)
                - processed box into numpy array

            Here is an example of a processed sample (with a single label):

            {'image_file_name': 'b1c81faa-3df17267.jpg',
             'image_file_path':
               '/path/to/bdd100k/images/100k/val/b1c81faa-3df17267.jpg',
             'image_hw': (720, 1280),
             'weather': 'clear',
             'timeofday': 'night',
             'scene': 'highway',
             'labels': [{'id': '34',
               'category': 'car',
               'occluded': False,
               'truncated': False,
               'trafficLightColor': 'NA',
               'box': array([819.464053, 280.082505, 889.23726 , 312.742305])}
             ]}
        """
        processed_samples = []

        for sample in samples:
            image_file_path = f"{self.images_path}/{sample['name']}"
            processed_sample = {
                'image_file_name': sample['name'],
                'image_file_path': image_file_path,
                'height': 720,
                'width': 1280,
                **sample['attributes'],  # Unpack image-level attributes.
            }


            # Process labels.
            labels = []
            for label in sample['labels']:
                processed_label = {
                    'id': label['id'],
                    'category': label['category'],
                    **label['attributes'],  # Unpack label-level attributes.
                }
                # Process box: dict[str, float] --> np.ndarray
                box = label['box2d']
                processed_label['box'] = np.array([
                    box['x1'],
                    box['y1'],
                    box['x2'],
                    box['y2'],
                ])
                labels.append(processed_label)

            # Add processed sample to collection.
            processed_sample['labels'] = labels
            processed_samples.append(processed_sample)

        return processed_samples

    @staticmethod
    def imread(image_path):
        """Read an image file and return it as a (H, W, 3) np.uint8 ndarray"""
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        #return np.moveaxis(image, -1, 0)
        return image

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index: int):
        sample = self._labels[index]
        sample['image'] = self.imread(sample['image_file_path'])
        return sample
