#/usr/bin/python3

'''
@author: Guocong Song
'''
import subprocess
import sys
from enum import Enum
import os

VW = os.path.join(os.environ['VW_BIN'], 'vw')

def quadratic(a, excd=[]):
    b = []
    def combine(a, i):
        if i == len(a):
            return
        for x in a[i:]:
            pair = ''.join([a[i], x])
            if pair in excd:
                continue
            b.append(' '.join(['-q', pair]))
        combine(a, i + 1)
    combine(a, 0)
    return b


class Option(Enum):
    poly_1 = 2
    poly_2 = 3
    quad_11 = 11
    quad_12 = 12
    quad_13 = 13
    poly_3 = 100
    
    @classmethod
    def fromstring(cls, str):
        return getattr(cls, str, None)
    
    @classmethod
    def features(cls, val):
        return ' --l2 2e-8 -l 0.01'
        #return '--stage_poly --batch_sz 800000 --batch_sz_no_doubling --sched_exponent 1.96' \
#        return ' --l2 2e-8 -l 0.1 --ftrl_alpha 0.01 --ftrl_beta 1e-5 --ftrl'


# From original file:
  #      if val == Option.quad_11:
#  #          return ' '.join(quadratic(list('pnabcdg'), excd=['aa', 'bb', 'cc', 'gg'])) \
 #               + ' --l2 2e-8 -l 0.01 -q m:'
#                + ' --save_resume --l2 10e-8 --l1 1.3e-8 -l 0.007 -q m:'
 #       elif val == Option.quad_12:
 #           return ' '.join(quadratic(list('pnabcdg'), excd=['aa', 'bb', 'cc', 'gg'])) \
 #               + ' --save_resume --feature_mask log_bin.model --l2 2e-8 -l 0.007 -q m:'
#                + ' --save_resume --feature_mask log_bin.model --l2 10e-8 --l1 0e-8  -l 0.005 -q m:'
#        elif val == Option.quad_13:
 #           return ' '.join(quadratic(list('pnabcdg'), excd=['aa', 'bb', 'cc', 'gg'])) \
 #               + ' --save_resume --feature_mask log_bin.model --l2 2e-8 -l 0.005 -q m:'
#                + ' --save_resume --feature_mask log_bin.model --l2 10e-8 --l1 1.3e-8 -l 0.007 -q m:'
#        elif val == Option.poly_1:
#            return '--stage_poly --batch_sz 800000 --batch_sz_no_doubling --sched_exponent 1.96' \
#                + ' --save_resume --l2 6e-8 --l1 1.2e-8 -l 0.007'
#        elif val == Option.poly_2:
#            return '--stage_poly --batch_sz 4000000 --batch_sz_no_doubling --sched_exponent 2.3' \
#                + ' --save_resume --l2 6e-8 --l1 1.2e-8 -l 0.007'
#        elif val == Option.poly_3:
#            return '--stage_poly --batch_sz 7500000 --batch_sz_no_doubling --sched_exponent 2.5' \
#                + ' --save_resume --l2 6e-8 --l1 1.2e-8 -l 0.007'
 #       else:
#            print('wrong mode:', val)
#            sys.exit(1)


def train(fname, option, passes, shfl_win):
    feeder = 'zcat %s | cut -f2 | /usr/bin/python3 ../scripts/shuffle.py %s | ' % (fname, shfl_win)
#From original file: feeder = 'zcat %s | ' % (fname)  #cut -f2 | /usr/bin/python3 ../scripts/shuffle.py %s | ' 
    loss = '--loss_function logistic'
    bits =  '-b 27'
    #-0.4878
    pass_args = '--passes %s --holdout_off -C -0.4878' % passes
    features = Option.features(option)
#From original file:    update = '--adaptive --invariant --power_t 0.5'
#    update = '--power_t 0.2 --ftrl'
    update = '--adaptive --invariant --power_t 0.25'
    io = '-c --compressed -f log_bin.model -k'
    command = ' '.join([feeder, VW, loss, bits, pass_args, features, update,io])
    print(command)
    subprocess.call(command, stdout=sys.stdout, shell=True, executable='/bin/bash')
    

def test(fname, part=''):
    predictions = 'prediction_' + fname.split('.')[0] +'_' + part + '.txt'
    predictor = 'zcat %s | cut -f2 | %s -t -i log_bin.model -p %s --quiet' % (fname, VW, predictions)
    subprocess.call(predictor, stdout=sys.stdout, shell=True, executable='/bin/bash')
    
    
if __name__ == '__main__':
    option = Option.fromstring(sys.argv[1])
    passes = sys.argv[2]
    shfl_win = sys.argv[3]
    
    train('train.vw.gz', option, passes, shfl_win)
    test('test.vw.gz', str(option))
    
