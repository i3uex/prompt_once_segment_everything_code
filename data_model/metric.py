import logging
from collections import namedtuple
from enum import Enum

log = logging.getLogger(__name__)

MetricItem = namedtuple(
    typename='MetricItem',
    field_names=['code', 'description', 'column_name']
)


class Metric(MetricItem, Enum):
    """
    Possible metrics to evaluate the segmentation process.
    """

    JaccardIndex = MetricItem(
        code='jaccard',
        description='Jaccard Index',
        column_name='jaccard'
    )
    DiceScore = MetricItem(
        code='dice',
        description='Dice Score',
        column_name='dice'
    )
    SAMScore = MetricItem(
        code='sam_score',
        description='SAM\'s Score',
        column_name='sam_score'
    )
    NotImplemented = MetricItem(
        code='',
        description='',
        column_name=''
    )

    @classmethod
    def from_string(
            cls,
            code: str
    ):
        """
        Get an instance of this class from its code as a string.

        :param code: value of the code as a string.

        :return: instance of this enumeration.
        """

        log.info('Create instance of Metric from its code')
        log.debug(f'Metric.from_string('
                  f'code="{code}")')

        for metric in list(Metric):
            if code.lower() == metric.code.lower():
                return metric

        return cls.NotImplemented
