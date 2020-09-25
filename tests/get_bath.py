"""测试获取bath轨道的功能"""


import numpy
from basics.dummy import BathOrbital, OneRDM, HoppingOperator, UMatrix
from rhf import RHFConfig


def bath1():
    '''获取bath'''
    #比如一个一维链，现在先创建一个hopping矩阵
    lsize = 6
    hopping = numpy.zeros([lsize, lsize], dtype=float)
    for orb in range(lsize - 1):
        hopping[orb, orb+1] = -1.0
        hopping[orb+1, orb] = -1.0
    hopping[0, lsize-1] = -1.0
    hopping[lsize-1, 0] = -1.0
    #
    hop = HoppingOperator(hopping)
    rhfcfg = RHFConfig(hop, 0.0, lsize//2)
    print(hop)
    print(rhfcfg.rhf_energy)
    #
    umat = UMatrix(lsize)
    rdm = OneRDM('D^mf', lsize//2, rhfcfg.rhf_fockop, umat)
    print(rdm.val)
    print(numpy.linalg.eigvalsh(rdm.val))
    bo1 = BathOrbital(rdm, [1, 2])
    print(bo1)


if __name__ == "__main__":
    bath1()
