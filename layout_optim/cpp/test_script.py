import eigen_dist_test
import numpy as np
from time import time

SIZE = 1000

t1 = time()
x = eigen_dist_test.test_dist_repl(SIZE)
t2 = time()
print('time vector replication: ', t2-t1)

t1 = time()
x = eigen_dist_test.test_dist_2mat(SIZE)
t2 = time()
print('time 2mat: ', t2-t1)

t1 = time()
x = eigen_dist_test.test_dist_loop(SIZE)
t2 = time()
print('time loop: ', t2-t1)


# print(np.round(x,2))

