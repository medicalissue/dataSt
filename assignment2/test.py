import numpy as np
import matplotlib.pyplot as plt
import time
import random

np.random.seed(20240454)

# 1-1
def assign(La):
    return La.copy()

La = [1]
Lb = assign(La)
La[0] = 0
print(f"Lb[0] = {Lb[0]}")

# 1-2
def listDeletionTime(n):
    times = np.zeros(1000)
    for i in range(1000):
        li = list(range(n))
        start = time.perf_counter()
        while li:
            li.pop(0)
        times[i] = time.perf_counter() - start
    return times

def arrayDeletionTime(n):
    times = np.zeros(1000)
    for i in range(1000):
        arr = np.arange(n)
        start = time.perf_counter()
        arr = np.delete(arr, np.arange(len(arr)))
        times[i] = time.perf_counter() - start
    return times

n = 100
listDelTime = listDeletionTime(n)
arrayDelTime = arrayDeletionTime(n)

plt.hist(listDelTime, bins=100)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig("./histListDel")
plt.close()

plt.hist(arrayDelTime, bins=100)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig("./histArrayDel")
plt.close()

ns = [100, 300, 500, 700, 900]
ratios = []
for n in ns:
    listT = listDeletionTime(n)
    arrayT = arrayDeletionTime(n)
    ratio = listT / arrayT
    ratios.append([ratio.min(), ratio.mean(), ratio.max()])
ratios = np.array(ratios)
plt.plot(ns, ratios[:,0], label="min")
plt.plot(ns, ratios[:,1], label="avg")
plt.plot(ns, ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./ListVsArrayDel")
plt.close()

# 1-3
def listConcatTime(n):
    times = np.zeros(1000)
    A = list(range(n))
    B = list(range(n))
    for i in range(1000):
        start = time.perf_counter()
        A + B
        times[i] = time.perf_counter() - start
    return times

def arrayConcatTime(n):
    times = np.zeros(1000)
    A = np.arange(n)
    B = np.arange(n)
    for i in range(1000):
        start = time.perf_counter()
        np.concatenate((A, B))
        times[i] = time.perf_counter() - start
    return times

n = 100
listConTime = listConcatTime(n)
arrayConTime = arrayConcatTime(n)

plt.hist(listConTime, bins=100)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig("./histListCon")
plt.close()

plt.hist(arrayConTime, bins=100)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.xscale("log", base=2)
plt.yscale("log")
plt.savefig("./histArrayCon")
plt.close()

ratios = []
for n in ns:
    listT = listConcatTime(n)
    arrayT = arrayConcatTime(n)
    ratio = listT / arrayT
    ratios.append([ratio.min(), ratio.mean(), ratio.max()])
ratios = np.array(ratios)
plt.plot(ns, ratios[:,0], label="min")
plt.plot(ns, ratios[:,1], label="avg")
plt.plot(ns, ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./ListVsArrayCon")
plt.close()

# 2

# 2-1-a
def PolyList(f):
    deg = len(f) - 1
    return [[deg - i, c] for i, c in enumerate(f) if c != 0]

g = [3, 0, 0, 0, 0, 0, 0, 2, 0]
print(f"When g(x) = 3x ^ 8 + 2x, PolyList(g) = {PolyList(g)}")

# 2-1-b
def PolyEvalList(f, c):
    t = PolyList(f)
    ret = 0
    for exp, coef in t:
        ret += coef * (c ** exp)
    return ret

def PolyEvalArray(f, c):
    return np.polyval(f, c)

fx = [1] + [0]*48 + [-1, 0]
ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyEvalList(fx, 0.5)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    PolyEvalArray(fx, 0.5)
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyEval")
plt.close()

# 2-1-c
def PolyAddList(f, g):
    if len(f) < len(g):
        f = [0] * (len(g) - len(f)) + f
    else:
        g = [0] * (len(f) - len(g)) + g
    return [a + b for a, b in zip(f, g)]

def PolyAddArray(f, g):
    if len(f) < len(g):
        f = np.pad(f, (len(g)-len(f), 0))
    else:
        g = np.pad(g, (len(f)-len(g), 0))
    return f + g

fx = [1] + [0]*28 + [-1, 1]
gx = [3] + [0]*5 + [2, 0]
ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyAddList(fx, gx)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    PolyAddArray(np.array(fx), np.array(gx))
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyAdd")
plt.close()

# 2-1-d
def PolyProdList(f, g):
    ret = [0] * (len(f) + len(g) - 1)
    for i in range(len(f)):
        for j in range(len(g)):
            ret[i + j] += f[i] * g[j]
    return ret

def PolyProdArray(f, g):
    return np.convolve(f, g)

fx = [1, -1, 0, -1, 1]
gx = [3, 0, 0, -4, 2, 0]
ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyProdList(fx, gx)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    PolyProdArray(np.array(fx), np.array(gx))
    t2 = time.perf_counter() - start
    ratios[i] = t1 / t2

plt.hist(ratios, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyProd")
plt.close()

# 2-1-e
def PolyIntList(f, c):
    n = len(f)
    ret = [f[i] / (n - i) for i in range(n)]
    ret.append(c)
    return ret

def PolyIntArray(f, c):
    n = len(f)
    ret = np.array([f[i] / (n - i) for i in range(n)])
    return np.append(ret, c)

fx = [1, -1, -1, 0]
c = 5
ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyIntList(fx, c)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    PolyIntArray(fx, c)
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2

plt.hist(ratios, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyInt")
plt.close()

# 2-1-f
def PolyDiffList(f):
    n = len(f)
    return [f[i] * (n - i - 1) for i in range(n - 1)]

def PolyDiffArray(f):
    n = len(f)
    deg = np.arange(n - 1, 0, -1)
    return f[:-1] * deg

fx = [4] + [0]*2 + [2, 6] + [0]*1 + [-1] + [0]*3 + [-1]
ratios = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyDiffList(fx)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    PolyDiffArray(np.array(fx))
    t2 = time.perf_counter() - start

    ratios[i] = t1 / t2
plt.hist(ratios, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyDiff")
plt.close()

# 2-2-a
def SharingList(A):
    return [list(map(int, np.where(A[:, i])[0])) for i in range(A.shape[1])]

A = np.random.binomial(1, 0.2, size=(20, 10))
print(f"SharingList(A) = {SharingList(A)}")

# 2-2-b
def FindPopularList(A_list):
    freq = {}
    for sublist in A_list:
        for item in sublist:
            freq[item] = freq.get(item, 0) + 1

    return max(freq, key=freq.get)


def FindPopularArray(A):
    return int(np.argmax(np.sum(A, axis=1)))

ratios = np.zeros(1000)
for i in range(1000):
    A_temp = np.random.binomial(1, 0.2, size=(20, 10))
    A_list_temp = SharingList(A_temp)
    
    start = time.perf_counter()
    FindPopularList(A_list_temp)
    t1 = time.perf_counter() - start
    
    start = time.perf_counter()
    FindPopularArray(A_temp)
    t2 = time.perf_counter() - start
    
    ratios[i] = t1 / t2
plt.hist(ratios, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistSharing")
plt.close()

# 3-1-a
def generateOrderedSet(n):
    return np.sort(np.random.randint(0, 101, n))

def Member(A, e):
    if len(A) == 0:
        return False
    p = 0
    while True:
        a = A[p]
        if a < e:
            if p == len(A) - 1:
                return False
            else:
                p += 1
        elif a > e:
            return False
        else:
            return True

def Subset(A, B):
    if len(A) == 0:
        return True
    p = 0
    while True:
        if Member(B, A[p]):
            if p == len(A) - 1:
                return True
            else:
                p += 1
        else:
            return False

def SubsetFast(A, B):
    i = j = 0
    while i < len(A) and j < len(B):
        if A[i] < B[j]:
            return False
        elif A[i] > B[j]:
            j += 1
        else:
            i += 1
            j += 1
    return i == len(A)

sizes = [10, 30, 50, 70, 90]
ratios = []
for n in sizes:
    temp = []
    for _ in range(1000):
        B_set = generateOrderedSet(n)
        start = time.perf_counter()
        Subset([0, 9], B_set)
        t1 = time.perf_counter() - start
        start = time.perf_counter()
        SubsetFast([0, 9], B_set)
        t2 = time.perf_counter() - start
        temp.append(t1 / t2)
    temp = np.array(temp)
    ratios.append([temp.min(), temp.mean(), temp.max()])
ratios = np.array(ratios)
plt.plot(sizes, ratios[:,0], label="min")
plt.plot(sizes, ratios[:,1], label="avg")
plt.plot(sizes, ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./Subset")
plt.close()

# 3-1-b
A_union = [0, 1, 4, 5, 8, 9, 10]

def UnionNon(A, B):
    i, j = 0, 0
    ret = []
    while i < len(A) and j < len(B):
        if A[i] < B[j]:
            ret.append(A[i])
            i += 1
        elif A[i] > B[j]:
            ret.append(B[j])
            j += 1
        else:
            ret.append(A[i])
            i += 1
            j += 1
    while i < len(A):
        ret.append(A[i])
        i += 1
    while j < len(B):
        ret.append(B[j])
        j += 1
    return ret

def Union(A, B):
    for b in B:
        if b not in A:
            A.append(b)
    return sorted(A)

ratios = []
for n in sizes:
    temp = []
    for _ in range(1000):
        B_union = generateOrderedSet(n)
        start = time.perf_counter()
        UnionNon(A_union, B_union)
        t1 = time.perf_counter() - start
        A_copy = A_union.copy()
        start = time.perf_counter()
        Union(A_copy, B_union)
        t2 = time.perf_counter() - start
        temp.append(t1 / t2)
    temp = np.array(temp)
    ratios.append([temp.min(), temp.mean(), temp.max()])
ratios = np.array(ratios)
plt.plot(sizes, ratios[:,0], label="min")
plt.plot(sizes, ratios[:,1], label="avg")
plt.plot(sizes, ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./Union")
plt.close()

# 3-1-c
def IntersectNon(A, B):
    i, j = 0, 0
    ret = []
    while i < len(A) and j < len(B):
        if A[i] < B[j]:
            i += 1
        elif A[i] > B[j]:
            j += 1
        else:
            ret.append(A[i])
            i += 1
            j += 1
    return ret

def Intersect(A, B):
    return [a for a in A if a in B]

ratios = []
for n in sizes:
    temp = []
    for _ in range(1000):
        B_intersect = generateOrderedSet(n)
        start = time.perf_counter()
        IntersectNon(A_union, B_intersect)
        t1 = time.perf_counter() - start
        A_copy = A_union.copy()
        start = time.perf_counter()
        Intersect(A_copy, B_intersect)
        t2 = time.perf_counter() - start
        temp.append(t1 / t2)
    temp = np.array(temp)
    ratios.append([temp.min(), temp.mean(), temp.max()])
ratios = np.array(ratios)
plt.plot(sizes, ratios[:,0], label="min")
plt.plot(sizes, ratios[:,1], label="avg")
plt.plot(sizes, ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./Intersect")
plt.close()