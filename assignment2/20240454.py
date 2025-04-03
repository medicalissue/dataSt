import numpy as np
import matplotlib.pyplot as plt
import time

#==============1-1==============
def assign(La):
    return La.copy()

La = [1]
Lb = assign(La)
La[0] = 0
print(Lb[0])

#==============1-2==============
LT, AT = list(), list()
for _ in range(1000):
    li = list(range(100))
    arr = np.arange(100)
    
    start = time.perf_counter()
    while li:
        li.pop()
    LT.append(time.perf_counter() - start)
    
    start = time.perf_counter()
    arr = np.delete(arr, np.s_[:])
    AT.append(time.perf_counter() - start)
    
plt.figure()
plt.hist(LT, bins=500)
plt.xlabel('Time')
plt.ylabel('Freq')
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig('./histListDel')
plt.clf()

plt.figure()
plt.hist(AT, bins=500)
plt.xlabel('Time')
plt.ylabel('Freq')
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig('./histArrayDel')
plt.clf()


minTime = list()
maxTime = list()
avgTime = list()
for n in range(100, 901, 200):
    LT = np.zeros(1000)
    AT = np.zeros(1000)
    for i in range(1000):
        li = list(range(n))
        arr = np.arange(n)
        
        start = time.perf_counter()
        while li:
            li.pop()
        LT[i] = time.perf_counter() - start
        
        start = time.perf_counter()
        arr = np.delete(arr, np.s_[:])
        AT[i] = time.perf_counter() - start

    minTime.append((LT / AT).min())
    maxTime.append((LT / AT).max())
    avgTime.append((LT / AT).mean())
    
nSize = [str(i) for i in range(100, 901, 200)]    

plt.plot(nSize, minTime, marker='o', label='Min Time')
plt.plot(nSize, maxTime, marker='s', label='Max Time')
plt.plot(nSize, avgTime, marker='^', label='Average Time')
plt.xlabel("n")
plt.ylabel("ratio")
# plt.xscale("log", base=2)
# plt.yscale("log")
plt.legend()
plt.savefig("./ListVsArrayDel")

plt.clf()

