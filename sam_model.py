"""
Enumeration to simplify SAM usage and SAM 2 usage.

Each item contains the name SAM understands and corresponding path to its
weights. This way, given an instance of the enumeration the rest of needed
values are available. Besides, there is no possible misspelling as the
enumeration values are provided to the developer.

Items are sorted from the smallest (ViT_B) to the biggest (ViT_H). The biggest
is the more capable version of the model, but also the one that needs more
resources.

Four additional models are available: Tiny, Small, BasePlus and Large. These
are the models used in the SAM 2 paper. They are also sorted from the smallest
to the biggest.
"""

from collections import namedtuple
from enum import Enum
from pathlib import Path

SamModelItem = namedtuple(
    'SamModelItem',
    ['version', 'name', 'checkpoint', 'configuration', 'description']
)


class SamModel(SamModelItem, Enum):
    """
    SAM available models.
    """

    ViT_B = SamModelItem(
        version=1,
        name='vit_b',
        checkpoint=str(Path(__file__).parent / 'model_checkpoints/sam_vit_b_01ec64.pth'),
        configuration='',
        description='SAM ViT-B'
    )
    ViT_L = SamModelItem(
        version=1,
        name='vit_l',
        checkpoint=str(Path(__file__).parent / 'model_checkpoints/sam_vit_l_0b3195.pth'),
        configuration='',
        description='SAM ViT-L'
    )
    ViT_H = SamModelItem(
        version=1,
        name='vit_h',
        checkpoint=str(Path(__file__).parent / 'model_checkpoints/sam_vit_h_4b8939.pth'),
        configuration='',
        description='SAM ViT-H'
    )
    Tiny = SamModelItem(
        version=2,
        name='tiny',
        checkpoint=str(Path(__file__).parent / 'model_checkpoints/sam2_hiera_tiny.pt'),
        configuration='sam2_hiera_t.yaml',
        description='SAM 2 Tiny'
    )
    Small = SamModelItem(
        version=2,
        name='small',
        checkpoint=str(Path(__file__).parent / 'model_checkpoints/sam2_hiera_small.pt'),
        configuration = 'sam2_hiera_s.yaml',
        description='SAM 2 Small'
    )
    BasePlus = SamModelItem(
        version=2,
        name='base_plus',
        checkpoint=str(Path(__file__).parent / 'model_checkpoints/sam2_hiera_base_plus.pt'),
        configuration='sam2_hiera_b+.yaml',
        description='SAM 2 Base+'
    )
    Large = SamModelItem(
        version=2,
        name='large',
        checkpoint=str(Path(__file__).parent / 'model_checkpoints/sam2_hiera_large.pt'),
        configuration='sam2_hiera_l.yaml',
        description='SAM 2 Large'
    )
