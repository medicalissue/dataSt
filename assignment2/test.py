import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(20240454)

#==============1-1==============
def assign(La):
    return La.copy()

La = [1]
Lb = assign(La)
La[0] = 0
print(Lb[0])

#==============1-2==============
def listDeletionTime(n):
    times = np.zeros(1000)
    for i in range(1000):
        L = list(range(n))
        start = time.perf_counter()
        L.pop(0)
        times[i] = time.perf_counter() - start
    return times

def arrayDeletionTime(n):
    times = np.zeros(1000)
    for i in range(1000):
        A = np.arange(n)
        start = time.perf_counter()
        np.delete(A, 0)
        times[i] = time.perf_counter() - start
    return times

n = 100
listTime = listDeletionTime(n)
arrayTime = arrayDeletionTime(n)

plt.hist(listTime, bins=50)
plt.xlabel('Time')
plt.ylabel('Freq')
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig("./histListDel")
plt.clf()

plt.hist(arrayTime, bins=50)
plt.xlabel('Time')
plt.ylabel('Freq')
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig("./histArrayDel")
plt.clf()

ratios = []
ns = [100, 300, 500, 700, 900]
for n in ns:
    listT = listDeletionTime(n)
    arrayT = arrayDeletionTime(n)
    ratio = listT / arrayT
    ratios.append([ratio.min(), ratio.mean(), ratio.max()])

ratios = np.array(ratios)
plt.plot(ns, ratios[:,0], label='min')
plt.plot(ns, ratios[:,1], label='avg')
plt.plot(ns, ratios[:,2], label='max')
plt.legend()
plt.savefig("./ListVsArrayDel")
plt.clf()

#==============1-3==============
def listConcatTime(n):
    times = np.zeros(1000)
    A = list(range(n))
    B = list(range(n))
    for i in range(1000):
        start = time.perf_counter()
        _ = A + B
        times[i] = time.perf_counter() - start
    return times

def arrayConcatTime(n):
    times = np.zeros(1000)
    A = np.arange(n)
    B = np.arange(n)
    for i in range(1000):
        start = time.perf_counter()
        _ = np.concatenate((A, B))
        times[i] = time.perf_counter() - start
    return times

n = 100
listTime = listConcatTime(n)
arrayTime = arrayConcatTime(n)

plt.hist(listTime, bins=50, alpha=0.5, label='list')
plt.hist(arrayTime, bins=50, alpha=0.5, label='array')
plt.legend()
plt.savefig("./histListCon")
plt.clf()

ratios = []
for n in ns:
    listT = listConcatTime(n)
    arrayT = arrayConcatTime(n)
    ratio = listT / arrayT
    ratios.append([ratio.min(), ratio.mean(), ratio.max()])

ratios = np.array(ratios)
plt.plot(ns, ratios[:,0], label='min')
plt.plot(ns, ratios[:,1], label='avg')
plt.plot(ns, ratios[:,2], label='max')
plt.legend()
plt.savefig("./ListVsArrayCon")
plt.clf()

#==============2-1==============
def polyList(f):
    deg = len(f) - 1
    return [[deg - i, c] for i, c in enumerate(f) if c != 0]

g = [3] + [0]*6 + [2, 0]
print(polyList(g))

def polyEvalList(f, c):
    result = 0
    for coef in f:
        result = result * c + coef
    return result

def polyEvalArray(f, c):
    return np.polyval(f, c)

f = [1] + [0]*48 + [-1, 0]
ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    polyEvalList(f, 0.5)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    polyEvalArray(np.array(f), 0.5)
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=50)
plt.savefig("./histPolyEval")
plt.clf()
   
def polyAddList(f, g):
    if len(f) < len(g):
        f = [0] * (len(g) - len(f)) + f
    else:
        g = [0] * (len(f) - len(g)) + g
    return [a + b for a, b in zip(f, g)]

def polyAddArray(f, g):
    if len(f) < len(g):
        f = np.pad(f, (len(g) - len(f), 0))
    else:
        g = np.pad(g, (len(f) - len(g), 0))
    return f + g

f = [1] + [0]*28 + [-1, 1]
g = [3] + [0]*5 + [2, 0]

ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    polyAddList(f, g)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    polyAddArray(np.array(f), np.array(g))
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=50)
plt.savefig("./histPolyAdd")
plt.clf()

def polyProdList(f, g):
    result = [0]*(len(f)+len(g)-1)
    for i in range(len(f)):
        for j in range(len(g)):
            result[i+j] += f[i] * g[j]
    return result

def polyProdArray(f, g):
    return np.convolve(f, g)

f = [1, -1, 0, -1, 1]
g = [3, 0, 0, -4, 2, 0]

ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    polyProdList(f, g)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    polyProdArray(np.array(f), np.array(g))
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=50)
plt.savefig("./histPolyProd")
plt.clf()

def polyIntList(f, c):
    n = len(f)
    result = [f[i] / (n - i) for i in range(n)]
    result.append(c)
    return result

def polyIntArray(f, c):
    n = len(f)
    result = np.array([f[i] / (n - i) for i in range(n)])
    return np.append(result, c)

f = [1, -1, -1]
c = 5
ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    polyIntList(f, c)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    polyIntArray(np.array(f), c)
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=50)
plt.savefig("./histPolyInt")
plt.clf()

def polyDiffList(f):
    n = len(f)
    return [f[i] * (n - i - 1) for i in range(n - 1)]

def polyDiffArray(f):
    n = len(f)
    deg = np.arange(n - 1, 0, -1)
    return f[:-1] * deg

f = [4] + [0]*2 + [2, 6] + [0]*1 + [-1] + [0]*2 + [-1]

ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    polyDiffList(f)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    polyDiffArray(np.array(f))
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=50)
plt.savefig("./histPolyDiff")
plt.clf()

#==============2-2==============
def sharingList(A):
    return [list(np.where(A[:, i])[0]) for i in range(A.shape[1])]

A = np.random.binomial(1, 0.2, size=(20, 10))
print(sharingList(A))

def findPopularList(A_list):
    from collections import Counter
    flat = [item for sublist in A_list for item in sublist]
    return Counter(flat).most_common(1)[0][0]

def findPopularArray(A):
    return np.argmax(np.sum(A, axis=1))

ratios = np.zeros(1000)
for i in range(1000):
    A = np.random.binomial(1, 0.2, size=(20, 10))
    A_list = sharingList(A)

    start = time.perf_counter()
    findPopularList(A_list)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    findPopularArray(A)
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=50)
plt.savefig("./histSharing")
plt.clf()

#==============3-1==============
import random
random.seed(0)

def member(A, e):
    for a in A:
        if a == e:
            return True
        if a > e:
            return False
    return False

def subset(A, B):
    return all(member(B, a) for a in A)

def subsetFast(A, B):
    from bisect import bisect_left
    return all(B[bisect_left(B, a)] == a if bisect_left(B, a) < len(B) else False for a in A)

def generateOrderedSet(n):
    return sorted(random.sample(range(101), n))

ratios = []
sizes = [10, 30, 50, 70, 90]
for n in sizes:
    A = [0, 9]
    temp = []
    for _ in range(1000):
        B = generateOrderedSet(n)
        start = time.perf_counter()
        subset(A, B)
        t1 = time.perf_counter() - start

        start = time.perf_counter()
        subsetFast(A, B)
        t2 = time.perf_counter() - start

        temp.append(t1 / t2 if t2 != 0 else 0)
    ratios.append([min(temp), sum(temp)/len(temp), max(temp)])

ratios = np.array(ratios)
plt.plot(sizes, ratios[:,0], label='min')
plt.plot(sizes, ratios[:,1], label='avg')
plt.plot(sizes, ratios[:,2], label='max')
plt.legend()
plt.savefig("./Subset")
plt.clf()
