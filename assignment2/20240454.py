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

#==============2-1==============
def PolyList(f):
    fRev = f[::-1]
    ans = [[i, fRev[i]] for i in range(len(fRev)) if fRev[i] != 0][::-1]
    return ans

def PolyArray(f):
    fRev = f[::-1]
    ans = np.array([[i, fRev[i]] for i in range(len(fRev)) if fRev[i] != 0][::-1])
    return ans

def PolyEvalList(f,c):
    ans = 0
    for i in f:
        ans += i[1] * (c ** i[0])
    return ans

def PolyEvalArray(f,c):
    ans = 0
    for i in f:
        ans += i[1] * (c ** i[0])
    return ans

pel = np.zeros(1000)
pea = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyEvalList([[50, 1], [1, -1]], 1 / 2)
    pel[i] = time.perf_counter() - start
    
    start = time.perf_counter()
    PolyEvalArray(np.array([[50, 1], [1, -1]]), 1 / 2)
    pea[i] = time.perf_counter() - start
    
plt.figure()
plt.hist(pel / pea, bins=500)
plt.xlabel('Time')
plt.ylabel('Freq')
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig('./HistPolyEval')
plt.clf()