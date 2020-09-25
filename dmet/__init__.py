"""实现dmet算法中的一些主要功能"""

import scipy.optimize
from basics.dummy import HoppingOperator, UMatrix, OneRDM, BathOrbital
from basics.dummy import HamiltonianEmb
from slover import exact_slove
from rhf import RHFConfig
from .nele import get_nele_diff


class DMETConfig():
    '''程序计算的句柄\n
    创建的时候只需要hopping matrix，还有相互做用的u的大小\n
    然后还需要pairnum，这个数字是正负电子的数量（乘以2才是总数）\n
    之后是杂质的格子，这个格子现在必须是有平移不变的\n
    '''
    def __init__(self, hopmat, coef_u, pairnum, implist):
        self._lattlen = hopmat.shape[0]
        self._hop = HoppingOperator(hopmat)
        self._coef_u = coef_u
        self._umat = UMatrix(self._lattlen)
        self._mu_glob = 0.0
        self._pairnum = pairnum
        self._implist = implist
        self._rhf_cfg = RHFConfig(self._hop, coef_u, pairnum)

    def update_umat(self):
        '''更新一次UMatrix'''
        #先根据rhf和现在的umat构造密度矩阵
        rdm = OneRDM('h', self._pairnum, self._rhf_cfg.rhf_fockop, self._umat)
        #然后获取imp上面的bath轨道
        bob = BathOrbital(rdm, self._implist)
        #创建high level的哈密顿量
        ham = HamiltonianEmb.create_hemb(self._hop, self._coef_u, bob)
        #现在需要找到合适的Mu
        appr_mu = scipy.optimize.newton(
            get_nele_diff, self._mu_glob,
            args=[ham, 2*self._pairnum]
        )
        self._mu_glob = appr_mu
        print(self._mu_glob)
        eng, one_rdm_ed = exact_slove(ham, self._mu_glob, 2*self._pairnum)
        print(eng, one_rdm_ed)

        
