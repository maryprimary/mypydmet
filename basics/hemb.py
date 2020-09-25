"""表示杂质中的high level hamiltonian"""

import numpy
from . import NamedTensor
from .operator import HoppingOperator
from .bath import BathOrbital


class OneParticlePart(NamedTensor):
    """整个哈密顿量的单个粒子成分\n
    这个成分就是fock算符，但是构成这个算符的U项和之前的\n
    fock算符不一样的是，这个U项是从环境，也就是非纠缠的\n
    部分计算出来的\n
    """
    def __init__(self, hop: HoppingOperator, coef_u, bob: BathOrbital):
        #先计算出来非纠缠部分的密度矩阵
        #有非纠缠部分的粒子的时候，bath orbital中的nointocc是有数值的
        #nointocc中的数值是0或者2，因为RHF上下是一样的
        nointdiag = numpy.diag(bob.nointocc)
        #把非纠缠的轨道的密度矩阵求出来，这时候已经乘过2了
        nointrdm = numpy.matmul(bob.val, numpy.matmul(nointdiag, bob.val.transpose()))
        #然后用hopping矩阵和非纠缠的密度矩阵求出fock算符
        hopping = hop.val
        interact = numpy.diag(
            0.5 * coef_u * numpy.diag(nointrdm)
        )
        fock_env = hopping + interact
        #然后要旋转这个fock算符，到相互作用轨道的基上
        rotmat = bob.interact_orbital
        fock_dmet = numpy.matmul(
            rotmat.transpose(), numpy.matmul(fock_env, rotmat) 
        )
        super().__init__(
            'h^x', fock_dmet.shape, ['k', 'l'],
            initv=fock_dmet
        )


class TwoParticlePart(NamedTensor):
    """整个哈密量的两粒子部分\n
    之前的单粒子中的U项是在环境中产生的，用密度矩阵算出来的\n
    ，这个时候的U项是impurity中产生的，用轨道自己算出来的\n
    hubbard模型中的U只在一个格子上其作用\n
    """
    def __init__(self, coef_u, bob: BathOrbital):
        #先拿出纠缠的轨道，这个的维度应该是（格子数，纠缠轨道数）
        intorb = bob.interact_orbital
        intdim = intorb.shape[1]
        #最后的结果应该是[intdim, intdim, intdim. intdim]
        dpp = coef_u * numpy.einsum('kp,kq,kr,ks->pqrs', intorb, intorb, intorb, intorb)
        #这个是不需要再做基的变换的
        super().__init__(
            '(pq|rs)', [intdim, intdim, intdim, intdim],
            ['p', 'q', 'r', 's'],
            initv=dpp
        )


class HamiltonianEmb():
    """这个是代表一个杂质区域的哈密顿量\n
    这个哈密顿量是完全在 2 * 杂质大小，这个大小的格子上的\n
    这里不找出具体的哈密顿量的表示，只是存储一些东西\n
    在创建的时候
    """
    def __init__(
            self,
            opp: OneParticlePart,
            tpp: TwoParticlePart,
            coef_u,
            hop: HoppingOperator,
            bob: BathOrbital
        ):
        self._coef_u = coef_u
        self._hop = hop
        self._tpp = tpp
        self._opp = opp
        self._bob = bob


    @staticmethod
    def create_hemb(
            hop: HoppingOperator,
            coef_u,
            bob: BathOrbital
        ):
        '''从hopping， u，还有bath轨道创建好哈密顿量'''
        opp = OneParticlePart(hop, coef_u, bob)
        tpp = TwoParticlePart(coef_u, bob)
        return HamiltonianEmb(opp, tpp, coef_u, hop, bob)


    @property
    def bath_orbital(self):
        '''bath轨道'''
        return self._bob

    @property
    def one_particle_part(self):
        '''哈密顿量中的单粒子成分'''
        return self._opp

    @property
    def two_particle_part(self):
        '''哈密顿量中的两粒子成分'''
        return self._tpp

    @property
    def hopping_operator(self):
        '''模型嗯的hopping矩阵'''
        return self._hop

    def __str__(self):
        template = 'HamiltonianEmb: \n\n'
        template += 'coef_u: %.3f\n\n' % self._coef_u
        template += str(self._bob) + '\n'
        template += str(self._hop) + '\n'
        template += str(self._opp) + '\n'
        template += str(self._tpp) + '\n'
        return template
