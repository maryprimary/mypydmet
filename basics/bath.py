"""和文章中bath轨道的实现有关的功能\n
[1] A Practical Guide to Density Matrix Embedding Theory in Quantum\n
Chemistry. JCTC. Sebastian Wouters\n
"""

from typing import List
import numpy
from . import NamedTensor, PRINT_LEVEL
from .rdm import OneRDM

class BathOrbital(NamedTensor):
    """主要是包含了Bath轨道的信息，还有一些其他会用到的数值\n
    需要整个格子上的密度矩阵，Impurity的位置\n
    """
    def __init__(self, denmat: OneRDM, implist: List[int]):
        self._denmat = denmat
        self._implist = implist
        #先得到除去impurity的格子的RDM
        block = self._get_emb1rdm()
        #然后按照[1]中2.2的描述，对角化这个block以后，
        #有一些本征值是0或者1的，是非纠缠的轨道，在这之间
        #的，是纠缠的轨道
        orbvecs, self._nointocc = self._get_sorted_eig(block)
        #
        super().__init__('A', self._denmat.val.shape, ['Mu', 'Nu'], initv=orbvecs)


    def _get_emb1rdm(self):
        '''从整体的rdm中找出没有impurity的部分（去掉不想要的行还有列）'''
        #denmat是整个格子上的
        emblist = [idx for idx in range(self._denmat.val.shape[0])\
            if idx not in self._implist]
        #从rdm中得到没有杂质轨道的项
        #直接[emblist, emblist]是不行的，只会得到对角项
        block = self._denmat.val[emblist, :]
        block = block[:, emblist]
        return block


    def _get_sorted_eig(self, block):
        '''将没有杂质的块对角化，找出非纠缠的几个轨道'''
        eigvals, eigvecs = numpy.linalg.eigh(block)
        #找出距离2和0最远的几个，这些是纠缠的轨道
        idx = numpy.argsort(numpy.abs(eigvals - 1.0))
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        #bath轨道的数量和impurity轨道的数量是一样的，前implist个都是
        #之后的就是非纠缠的轨道了
        nointorb = eigvecs[:, len(self._implist):]
        nointval = eigvals[len(self._implist):]
        #重新给非纠缠的轨道排序，让本正值大的在前面
        idx2 = numpy.argsort(nointval)[::-1]
        nointval = nointval[idx2]
        nointorb = nointorb[:, idx2]
        #重新给接回去
        eigvecs[:, len(self._implist):] = nointorb
        #现在需要重新把impurity轨道重新插入回去
        eigvecs2 = numpy.zeros(self._denmat.val.shape)
        eigvecs2[:len(self._implist), :len(self._implist)] =\
            numpy.eye(len(self._implist))
        eigvecs2[len(self._implist):, len(self._implist):] =\
            eigvecs
        #整理出一个非纠缠轨道的占据数，有0还有2，纠缠轨道用0来填充
        nointocc = numpy.zeros(self._denmat.val.shape[0])
        nointocc[2*len(self._implist):] = nointval
        return eigvecs2, nointocc


    @property
    def nointocc(self):
        '''非相互作用的轨道上面的粒子数'''
        return self._nointocc

    @property
    def interact_orbital(self):
        '''所有的纠缠轨道'''
        return self._val[:, :2*len(self._implist)]

    @property
    def implist(self):
        '''杂质格子的编号'''
        return self._implist

    def __str__(self):
        template = super().__str__()
        template += 'Bath Orbital Info: \n'
        template += 'Impurity List: %s\n' % str(self._implist)
        if PRINT_LEVEL > 0:
            template += 'PRINT LEVEL 1\n'
            template += 'Noninteraction Orbital Occpation: \n'
            template += str(self._nointocc)
            template += '\n'
        return template
