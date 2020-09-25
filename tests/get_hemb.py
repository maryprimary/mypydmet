"""测试获取hemb"""

import numpy
from basics.dummy import BathOrbital, OneRDM, HoppingOperator, UMatrix, HamiltonianEmb
from basics.hemb import OneParticlePart, TwoParticlePart
from rhf import RHFConfig
from slover import exact_slove
#from slover.exact import construct_full_hamiltonian

def hemb1():
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
    #
    umat = UMatrix(lsize)
    rdm = OneRDM('D^mf', lsize//2, rhfcfg.rhf_fockop, umat)
    bo1 = BathOrbital(rdm, [1, 2])
    #测试获取hemb
    opp = OneParticlePart(hop, 0.0, bo1)
    print(opp)
    #
    tpp = TwoParticlePart(1.0, bo1)
    print(tpp)
    #
    hemb = HamiltonianEmb.create_hemb(hop, 8.0, bo1)
    print(hemb)
    return hemb

def hemb2():
    '''测试构造完整的哈密顿量'''
    he1 = hemb1()
    #print(construct_full_hamiltonian(he1, 0.0))
    print(exact_slove(he1, 0.0, 6))


if __name__ == "__main__":
    hemb2()
