# abstract base class for assigner
# assigner is to match the gtboxes with bboxes, decide which are positive, 
# which are negtive, which are neglected


from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass
