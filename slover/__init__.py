"""进行精确求解，主要是要求1RDM和2RDM这两个\n
Reduced Density Matrix
"""

import numpy
from basics.dummy import HamiltonianEmb
from .exact import construct_full_hamiltonian, construct_2rdm

def exact_slove(hamiltonian: HamiltonianEmb, mu_add, nelec):
    '''使用全对角化得到结果\n
    最后返回能量还有密度矩阵\n
    '''
    #得到特定Mu的数值
    fullmat = construct_full_hamiltonian(hamiltonian, mu_add)
    #对角化结果
    eigvals, eigvecs = numpy.linalg.eigh(fullmat)
    #拿出基态，构造两粒子的密度矩阵
    ground = eigvecs[:, 0]
    print('f', fullmat)
    print('g', ground, numpy.dot(ground, ground))
    two_rdm = construct_2rdm(hamiltonian, ground)
    #print('t', two_rdm)
    #从两粒子的密度矩阵得到单粒子的密度矩阵
    #这个时候需要杂质中的电子数，从总的电子数减去环境电子数得到
    nelec_in_imp = int(round(nelec - numpy.sum(hamiltonian.bath_orbital.nointocc)))
    one_rdm = numpy.einsum('ikjk->ij', two_rdm) / (nelec_in_imp - 1)
    #print('rdiff', one_rdm / one_rdm2)
    #计算杂质的能量
    implen = len(hamiltonian.bath_orbital.implist)
    #需要两个hopping矩阵，一个包含u的矩阵，Fock算符中包含u和一个hopping
    #构造一个旋转到bath轨道上的hopping
    hopmat = hamiltonian.hopping_operator.val
    rotmat = hamiltonian.bath_orbital.interact_orbital
    hoprot = numpy.matmul(rotmat.transpose(), numpy.matmul(hopmat, rotmat))
    energy = 0.5 * numpy.einsum(
        'ij,ij->',
        #只需要杂质里面的几个格子的能量
        one_rdm[:implen, :],
        hoprot[:implen, :] +\
            hamiltonian.one_particle_part[:implen, :]
    )
    energy += 0.5 * numpy.einsum(
        'ijkl,ijkl->',
        two_rdm[:implen, :, :, :],
        hamiltonian.two_particle_part.val[:implen, :, :, :]
    )
    return energy, one_rdm
