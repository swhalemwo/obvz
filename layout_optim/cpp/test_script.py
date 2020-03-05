import eigen_dist_test
import numpy as np
from time import time

import matplotlib.pyplot as plt

# twomat_times = []
SIZE = 30

sizes = []
repl_times = []
repl_times2 = []
loop_openmp_times = []
loop_no_openmp_times = []

while SIZE < 400:
    
    # print(SIZE)

    t1 = time()
    x1 = eigen_dist_test.dist_loop_openmp(SIZE)
    t2 = time()
    loop_openmp_times.append(t2-t1)
    
    
    t1 = time()
    x2 = eigen_dist_test.test_dist_repl(SIZE)
    t2 = time()
    repl_times.append(t2-t1)

    
    t1 = time()
    x3 = eigen_dist_test.test_dist_repl2(SIZE)
    t2 = time()
    repl_times2.append(t2-t1)



    t1 = time()
    x4 = eigen_dist_test.dist_loop_no_openmp(SIZE)
    t2 = time()
    loop_no_openmp_times.append(t2-t1)


    # t1 = time()
    # x = eigen_dist_test.test_dist_2mat(SIZE)
    # t2 = time()
    # twomat_times.append(t2-t1)
    # print('time 2mat: ', t2-t1)

    sizes.append(SIZE)

    SIZE = int(SIZE * 1.1)

    

# hm would be nice to have graphs of speed test for different parameters
# print(np.round(x,2))

fig = plt.figure()
ax = plt.axes()

ax.plot(sizes, repl_times, label = 'repl')
ax.plot(sizes, repl_times2, label = 'repl2')
ax.plot(sizes, loop_no_openmp_times, label = 'loop_no_openmp')
ax.plot(sizes, loop_openmp_times, label = 'loop_openmp')


print('loop_open_mp mean: ', np.mean(loop_openmp_times))
print('loop_no_open_mp mean: ', np.mean(loop_no_openmp_times))
print('repl 1 mean: ', np.mean(repl_times))
print('repl 2 mean: ', np.mean(repl_times2))


# ratios = [i[0]/i[1] for i in zip(repl_times, loop_times)]
# ax.plot(sizes, ratios, label = 'ratios')
# print(np.mean(np.array(ratios)))

plt.legend()



# ax.plot(sizes, twomat_times)

plt.show()


