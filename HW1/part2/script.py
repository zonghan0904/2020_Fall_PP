import os

TIMES = 15

for i in range(TIMES):
    print("%d / %d"%(i + 1, TIMES))
    f = os.popen("./test_auto_vectorize -t 2")
