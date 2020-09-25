"""有关于Reduced Density Matrix的功能\n
\n
[1] A Practical Guide to Density Matrix Embedding Theory in Quantum\n
Chemistry. JCTC. Sebastian Wouters\n
"""

import numpy
from . import NamedTensor
from .operator import FockOperator
from .umat import UMatrix

class OneRDM(NamedTensor):
    '''用来构造1RDM\n
    需要提供这个RDM的名字，格子的大小（1RDM肯定是格子*格子）\n
    需要单粒子的哈密顿量h（这个其实就是hf近似的哈密顿量，[1]中14式）\n
    以及一个需要自适应计算的umat，这个u不是Hubbard模型中的umat\n
    还需要一共有多少粒子对（上下是一样的）\n
    '''
    def __init__(
            self, name, pairnum,
            fockop: FockOperator,
            umat: UMatrix,
        ):
        self._fockop = fockop
        self._umat = umat
        self._pairnum = pairnum
        denmat = self._slove_denmat()
        super().__init__(name, self._fockop.val.shape, ['k', 'l'], initv=denmat)


    def _slove_denmat(self):
        '''计算出现[1](14)这个哈密顿量对应的密度矩阵'''
        ham = self._fockop.val + self._umat.val
        eigvals, eigvecs = numpy.linalg.eigh(ham)
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        #
        occvecs = eigvecs[:, :self._pairnum]
        #构造密度矩阵，还是考虑上上下两个方向的，乘以2
        denmat = 2 * numpy.matmul(occvecs, occvecs.transpose())
        return denmat
