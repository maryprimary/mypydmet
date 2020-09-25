"""常用的功能"""

import numpy

#这个数值越大，打印的就应该越详细
PRINT_LEVEL = 1


class NamedTensor():
    '''只是包装一下numpy的数组\n
    需要提供这个数组本身的名字，各个分量的维度，角标等\n
    __getitem__和__setitem__直接使用的是ndarray的\n
    '''
    def __init__(self, name, dims, footnotes, initv=None):
        self._name = name
        self._dims = dims
        self._footnotes = footnotes
        #
        if initv is not None:
            self.set_val(initv)
        else:
            self._val = numpy.zeros(self._dims)
        #


    def set_val(self, initv):
        '''设置val'''
        if not isinstance(initv, numpy.ndarray):
            initv = numpy.array(initv)
        if not numpy.allclose(initv.shape, self._dims):
            raise ValueError('dims和initv大小不一致')
        self._val = initv.copy()


    @property
    def val(self) -> numpy.ndarray:
        '''ndarray数组'''
        return self._val

    def __getitem__(self, keys):
        return self._val[keys]

    def __setitem__(self, keys, values):
        self._val[keys] = values

    def __str__(self):
        footnote = ','.join(self._footnotes)
        template = 'Type: %s\n' % self.__class__.__name__
        template += 'Note: %s_{%s}\n' % (self._name, footnote)
        template += 'Dims: %s\n' % str(self._dims)
        if PRINT_LEVEL > 0:
            template += 'PRINT LEVEL 1\n'
            template += 'Val: \n'
            template += str(self._val)
            template += '\n'
        return template
