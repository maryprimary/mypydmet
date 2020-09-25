"""使用全对角化来得到精确的密度矩阵"""

import numpy
from basics.dummy import HamiltonianEmb


def _hopping_state(ridx, cidx, codelen):
    '''这个方法把单粒子的项在完整的空间中哪些位置有数值寻找出来\n
    ridx单粒子矩阵的行号，这个是目的编号，这个是在格子上的，cidx也是在格\n
    子上的，对应的是sidx变成二进制以后的位\n
    codelen是格子的长度
    '''
    space_dim = 2**codelen
    results = []
    for sidx in range(space_dim):
        code = list(bin(sidx)[2:].rjust(codelen, '0'))
        #如果第二个idx没有值或者第一个有值，就是0
        if code[cidx] == '0' or code[ridx] == '1':
            if cidx != ridx:
                continue
        #首先计算消灭cidx的符号
        sign = code[:cidx].count('1')
        #消灭掉cidx上面的1
        code[cidx] = '0'
        #然后加上产生ridx的符号
        sign += code[:ridx].count('1')
        #在ridx上产生1
        code[ridx] = '1'
        #最后生成的状态的编号
        eidx = int(''.join(code), 2)
        #
        sign = 1 if sign % 2 == 0 else -1
        results.append((eidx, sidx, sign))
    return results


def _interact_state(idx1, idx2, idx3, idx4, codelen):
    '''这个方法把连续四个升降算符在完整的空间中哪些位置有数值计算出来\n
    其中顺序是产生消灭产生消灭[1]（13），所以现在的1234，应该对应的是1324\n
    产生1先灭3产生2消灭4
    codelen是格子的长度\n
    '''
    space_dim = 2**codelen
    results = []
    for sidx in range(space_dim):
        code = list(bin(sidx)[2:].rjust(codelen, '0'))
        #一步一步判断是不是没有结果
        if code[idx4] == '0':
            continue
        sign = code[:idx4].count('1')
        code[idx4] = '0'
        if code[idx2] == '1':
            continue
        sign += code[:idx2].count('1')
        code[idx2] = '1'
        if code[idx3] == '0':
            continue
        sign += code[:idx3].count('1')
        code[idx3] = '0'
        if code[idx1] == '1':
            continue
        sign += code[:idx1].count('1')
        code[idx1] = '1'
        #最后生成的状态的编号
        eidx = int(''.join(code), 2)
        #
        sign = 1 if sign % 2 == 0 else -1
        results.append((eidx, sidx, sign))
    return results


def construct_hamiltonian_without_mu(hemb: HamiltonianEmb):
    '''构造哈密顿量，不包括控制粒子数的mu'''
    #首先把整个空间的基确定下来
    code_len = 2 * len(hemb.bath_orbital.implist)
    #按照'10101...'这种方式进行编号，将其转换成十进制就是在数组里面的index
    #整个空间的维度就是2^N，因为这个时候已经没有自旋上下的区别了
    #利用一个方向解出来结果，然后直接乘以2就可以了
    space_dim = numpy.power(2, code_len)
    #
    full_matrix = numpy.zeros([space_dim, space_dim])
    #现在的基都是在bath轨道上面的，先把单粒子对应的算符赋值过去
    hopping = hemb.one_particle_part.val
    for cidx in range(code_len):
        for ridx in range(code_len):
            val = hopping[ridx, cidx]
            ents = _hopping_state(ridx, cidx, code_len)
            for eidx, sidx, sign in ents:
                full_matrix[eidx, sidx] += sign * val
    #然后把两个粒子的算符赋值过去
    interact = hemb.two_particle_part.val
    for idx1 in range(code_len):
        for idx2 in range(code_len):
            for idx3 in range(code_len):
                for idx4 in range(code_len):
                    val = interact[idx1, idx2, idx3, idx4]
                    if val == 0:
                        continue
                    ents = _interact_state(idx1, idx2, idx3, idx4, code_len)
                    for eidx, sidx, sign in ents:
                        print(idx1, idx2, idx3, idx4, eidx, sidx)
                        full_matrix[eidx, sidx] += sign * val
    #raise
    #
    return full_matrix


def construct_full_hamiltonian(hemb: HamiltonianEmb, mu_add):
    '''得到完整的哈密顿量'''
    #没有Mu的部分是不需要重新做的
    mat_no_mu = getattr(hemb, '_exact_mat_no_mu', None)
    if mat_no_mu is None:
        mat_no_mu = construct_hamiltonian_without_mu(hemb)
        setattr(hemb, '_exact_mat_no_mu', mat_no_mu)
    #然后增加有Mu的部分
    code_len = 2 * len(hemb.bath_orbital.implist)
    space_dim = numpy.power(2, code_len)
    mat_with_mu = numpy.copy(mat_no_mu)
    for idx in range(space_dim):
        code = list(bin(idx)[2:2+len(hemb.bath_orbital.implist)].rjust(code_len, '0'))
        pnum = code.count('1')
        mat_with_mu[idx, idx] += mu_add * pnum
    return mat_with_mu


def construct_2rdm(hemb: HamiltonianEmb, ground):
    '''通过基态构造密度矩阵'''
    code_len = 2 * len(hemb.bath_orbital.implist)
    two_rdm = numpy.zeros([code_len, code_len, code_len, code_len])
    space_dim = numpy.power(2, code_len)
    for idx1 in range(code_len):
        for idx2 in range(code_len):
            for idx3 in range(code_len):
                for idx4 in range(code_len):
                    #在_interact_state中，就是按照1324的顺序进行生成的
                    #最后的结果也是应该放到1234中的
                    ents = _interact_state(idx1, idx2, idx3, idx4, code_len)
                    matop = numpy.zeros([space_dim, space_dim])
                    for eidx, sidx, sign in ents:
                        matop[eidx, sidx] = sign
                    val = numpy.dot(
                        ground.transpose(),
                        numpy.matmul(matop, ground)
                    )
                    two_rdm[idx1, idx2, idx3, idx4] = val
    return 2 * two_rdm


def test1():
    '''测试构成的哈密顿量'''
    #res = _hopping_state(3, 1, 6)
    #for eidx, sidx, sign in res:
    #    print(bin(eidx), bin(sidx), sign)
    #比如一个6长度的一维链，把单粒子的hopping matrix找到
    lsize = 6
    hopping = numpy.zeros([lsize, lsize], dtype=float)
    for orb in range(lsize - 1):
        hopping[orb, orb+1] = -1.0
        hopping[orb+1, orb] = -1.0
    hopping[0, lsize-1] = -1.0
    hopping[lsize-1, 0] = -1.0
    #从单粒子的hopping matrix找到整个one hot空间中的哈密顿量
    ham = numpy.zeros([2**lsize, 2**lsize], dtype=numpy.float)
    for cidx in range(lsize):
        for ridx in range(lsize):
            val = hopping[ridx, cidx]
            ents = _hopping_state(ridx, cidx, lsize)
            for eidx, sidx, sign in ents:
                ham[eidx, sidx] += sign * val
    print(ham)
    eigvals = numpy.linalg.eigvalsh(ham)
    print(eigvals * 2)
    #从两个粒子的相互作用找到one hot空间中的哈密顿量
    hamu = numpy.zeros([2**lsize, 2**lsize], dtype=numpy.float)
    for idx in range(lsize):
        ents = _interact_state(idx, idx, idx, idx, lsize)
        for eidx, sidx, sign in ents:
            hamu[eidx, sidx] += sign * 1.0
    print(hamu)


if __name__ == "__main__":
    test1()
