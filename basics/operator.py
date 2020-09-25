"""算符"""

#import numpy
from . import NamedTensor

class FockOperator(NamedTensor):
    '''用来保存fock operator\n
    需要提供这个数组本身的名字，维度\n
    fock operator本身是在格子的坐标上的\n
    ，角标用kl，肯定是正方\n
    '''
    def __init__(self, name, dim, initv=None):
        super().__init__(name, [dim, dim], ['k', 'l'], initv=initv)


class HoppingOperator(NamedTensor):
    '''用来保存hopping matrix\n
    需要提供完整的hopping矩阵，\n
    本身在格子上，脚标用kl\n
    '''
    def __init__(self, initv):
        super().__init__('T', initv.shape, ['k', 'l'], initv=initv)
