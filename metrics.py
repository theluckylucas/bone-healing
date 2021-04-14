import numpy
import medpy.metric.binary as mpm
from torch.nn.modules.loss import _Loss as LossModule
from torch.autograd import Variable


class BatchDiceLoss(LossModule):
    def __init__(self, label_weights, epsilon=0.0000001, dim=1):
        super(BatchDiceLoss, self).__init__()
        self._epsilon = epsilon
        self._dim = dim
        self._label_weights = label_weights
        print("DICE Loss weights classes' output by", label_weights)

    def forward(self, outputs, targets):
        assert not len(targets.shape)>3 or targets.shape[self._dim] == len(self._label_weights), \
            'Ground truth number of labels does not match with label weight vector'
        loss = 0.0
        for label in range(len(self._label_weights)):
            oflat = outputs.narrow(self._dim, label, 1).contiguous().view(-1)
            tflat = targets.narrow(self._dim, label, 1).contiguous().view(-1)
            assert oflat.size() == tflat.size()
            intersection = (oflat * tflat).sum()
            numerator = 2.*intersection + self._epsilon
            denominator = (oflat * oflat).sum() + (tflat * tflat).sum() + self._epsilon
            loss += self._label_weights[label] * (numerator / denominator)
        return 1.0 - loss
