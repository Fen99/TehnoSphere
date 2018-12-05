'''
@author: Guocong Song
'''
import pandas as pd
import sys
import numpy as np
import gzip


df = pd.read_csv(sys.stdin)
#p = 0.1 * df.p1 + 0.3 * df.p2 + 0.3 * df.p3 + 0.3 * df.p4
p = 1.0 * df.p1 #+ 1.0 * df.p2 #+ 0.15 * df.p3 + 0.15 * df.p4
df['Click'] = prob = 1.0 / (1.0 + np.exp(-p))

submission = 'submission.cvs'
print('saving to', submission, '...')
with open(submission, 'w') as f:
    df[['Id', 'Click']].to_csv(f, index=False)
