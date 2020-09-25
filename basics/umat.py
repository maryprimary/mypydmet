"""这个是需要自适应计算出来的哈密顿量中的一部分\n
[1] A Practical Guide to Density Matrix Embedding Theory in Quantum\n
Chemistry. JCTC. Sebastian Wouters\n
[1]中14\n
"""

from . import NamedTensor

class UMatrix(NamedTensor):
    '''这个umat主要在以后的自适应计算中使用\n
    大小就是格子大小,脚标kl，初始0\n
    '''
    def __init__(self, dim):
        super().__init__('U', [dim, dim], ['k', 'l'], initv=None)
