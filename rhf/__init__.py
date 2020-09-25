"""完成rhf运算的方法"""

import numpy
from basics.operator import HoppingOperator, FockOperator

class RHFConfig():
    '''RHF计算的句柄\n
    需要hopping矩阵,U还有粒子数，用来计算RHF的结果\n
    '''
    def __init__(self, top: HoppingOperator, coef_u, pairnum):
        self._top = top
        self._coef_u = coef_u
        self._pairnum = pairnum
        #
        self._hopping_denmat = None
        self._interact_mat = None
        self._rhf_energy = 0.
        self._rhf_fockop = None
        #解rhf
        self._slove()


    def _slove(self):
        '''解RHF'''
        #解出hopping矩阵的本征态，按照能量排序
        eigvals, eigvecs = numpy.linalg.eigh(self._top.val)
        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        #利用hopping矩阵的本征态构造一个密度矩阵
        occvecs = eigvecs[:, :self._pairnum]
        #在RHF里面向下和向上的是一样的，整体的密度应该乘以2，自旋上下是一样的，
        #需要对上下求和的时候直接乘2就行了
        self._hopping_denmat = 2 * numpy.matmul(occvecs, occvecs.transpose())
        #通过这个密度矩阵近似U项，这个时候之前乘的2又得除回来，因为上下是一项
        #没有对自旋上下的求和
        self._interact_mat = numpy.diag(
            0.5 * numpy.diag(self._hopping_denmat) * self._coef_u
        )
        #利用近似的密度矩阵求出哈密顿量的本正值
        #这时的hopping是上下都有的，不用乘0.5，U项需要，之前乘的0.5是构造近似的
        #U时候用的，现在还要乘一个（E = T_up + T_down + U)
        self._rhf_energy = numpy.multiply(
            self._top.val + 0.5 * self._interact_mat,
            self._hopping_denmat
        )
        self._rhf_energy = numpy.sum(self._rhf_energy)
        self._rhf_fockop = FockOperator(
            'h', self._top.val.shape[0],
            #这个时候存下来的就是一个自旋方向的
            initv=self._top.val + self._interact_mat
        )


    @property
    def rhf_energy(self):
        '''RHF的能量'''
        return self._rhf_energy

    @property
    def rhf_fockop(self):
        '''RHF计算得到的哈密顿量'''
        return self._rhf_fockop
