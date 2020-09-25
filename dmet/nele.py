"""计算某个Mu值的时候，Hemb中的电子数"""

import numpy
from basics.dummy import HamiltonianEmb
from slover import exact_slove

def get_nele_diff(
        mu_add,
        hemb: HamiltonianEmb,
        nelec
    ):
    eng, one_rdm = exact_slove(hemb, mu_add, nelec)
    #从one_rdm中得到粒子数
    if hemb.hopping_operator.val.shape[0] %\
        len(hemb.bath_orbital.implist) != 0:
        raise ValueError('只能用于能整除且有平移不变的Impurity')
    #一共的imp数目
    imp_number = hemb.hopping_operator.val.shape[0] // len(hemb.bath_orbital.implist)
    #print(eng * imp_number)
    #找到一个imp中的粒子数
    nele_imp =\
        numpy.trace(
            one_rdm[:len(hemb.bath_orbital.implist), :len(hemb.bath_orbital.implist)]
        )
    tot_nele = nele_imp * imp_number
    return  nelec - tot_nele
