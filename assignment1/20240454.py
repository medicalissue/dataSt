import numpy as np
import time
import matplotlib.pyplot as plt

#==============1-1==============

np.random.seed(20240454)

def prefixAverages1(X, n):
    A = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += X[j]
        A[i] = sum / (i + 1)
    return A

def prefixAverages2(X, n):
    A = np.zeros(n)
    sum = 0
    for i in range(n):
        sum += X[i]
        A[i] = sum / (i + 1)
    return A

n = 2 ** 6
slow = list()
optimized = list()
for _ in range(1000):
    X = np.random.uniform(0, 1, n)
    start = time.time()
    prefixAverages1(X, n)
    slow.append(time.time() - start)

    start = time.time()
    prefixAverages2(X, n)
    optimized.append(time.time() - start)

plt.hist(slow, color='red', label='slow')
plt.hist(optimized, color='green', label='optimized')
plt.legend()
plt.xscale("log")
plt.xlabel("Time")
plt.ylabel("Freq")
plt.savefig("./ratio_hist.jpg")

plt.clf()

minTime = list()
maxTime = list()
avgTime = list()
for i in range(6, 13):
    n = 2 ** i
    optimized = np.zeros(1000)
    for j in range(1000):
        X = np.random.uniform(0, 1, n)
        start = time.time()
        prefixAverages2(X, n)
        optimized[j] = time.time() - start
    minTime.append(optimized.min())
    maxTime.append(optimized.max())
    avgTime.append(optimized.mean())
    
nSize = [2 ** i for i in range(6, 13)]

plt.plot(nSize, minTime, marker='o', label='Min Time')
plt.plot(nSize, maxTime, marker='s', label='Max Time')
plt.plot(nSize, avgTime, marker='^', label='Average Time')
plt.xlabel("n")
plt.ylabel("time")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig("./ratio plot.jpg")

plt.clf()

#==============1-2==============

def movingAverages1(X, n, k):
    A = np.zeros(n)
    for i in range(n):
        if i < k - 1:
            A[i] = np.mean(X[:i + 1])
        else:
            A[i] = np.mean(X[i - k + 1:i + 1])
    return A

def movingAverages2(X, n, k):
    A = np.zeros(n)
    move = 0
    for i in range(n):
        move += X[i]
        if k <= i:
            move -= X[i - k]
        A[i] = move / min(i + 1, k)
    return A

n = 2 ** 6
k = 2 ** 5
slow = list()
optimized = list()
for _ in range(1000):
    X = np.random.uniform(0, 1, n)
    start = time.time()
    movingAverages1(X, n, k)
    slow.append(time.time() - start)

    start = time.time()
    movingAverages2(X, n, k)
    optimized.append(time.time() - start)

plt.hist(slow, color='red', label='slow')
plt.hist(optimized, color='green', label='optimized')
plt.legend()
plt.xscale("log")
plt.xlabel("Time")
plt.ylabel("Freq")
plt.savefig("./ratio_hist2.jpg")

plt.clf()

minTime = list()
maxTime = list()
avgTime = list()
for i in range(7, 12):
    n = 2 ** i
    optimized = np.zeros(1000)
    for j in range(1000):
        X = np.random.uniform(0, 1, n)
        start = time.time()
        movingAverages2(X, n, k)
        optimized[j] = time.time() - start
    minTime.append(optimized.min())
    maxTime.append(optimized.max())
    avgTime.append(optimized.mean())
    
nSize = [2 ** i for i in range(7, 12)]

plt.plot(nSize, minTime, marker='o', label='Min Time')
plt.plot(nSize, maxTime, marker='s', label='Max Time')
plt.plot(nSize, avgTime, marker='^', label='Average Time')
plt.xlabel("n")
plt.ylabel("time")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig("./ratio plot2.jpg")

plt.clf()

#==============1-3===============

def findMissing(A, n):
    return (n * (n - 1) / 2) - sum(A)

six = list()
n = 2 ** 6
for j in range(1000):
    A = list(range(n))
    np.random.shuffle(A)
    A.pop()
    start = time.time()
    findMissing(A, n)
    six.append(time.time() - start)
    
seven = list()
n = 2 ** 7
for j in range(1000):
    A = list(range(n))
    np.random.shuffle(A)
    A.pop()
    start = time.time()
    findMissing(A, n)
    seven.append(time.time() - start)

plt.subplot(121)
plt.hist(six, color='red', label='2**6')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Freq")

plt.subplot(122)
plt.hist(seven, color='green', label='2**7')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Freq")

plt.savefig("./ratio_hist3.jpg")
plt.clf()

minTime = list()
maxTime = list()
avgTime = list()
for i in range(7, 12):
    n = 2 ** i
    find = np.zeros(1000)
    for j in range(1000):
        X = np.random.uniform(0, 1, n)
        start = time.time()
        findMissing(X, n)
        find[j] = time.time() - start
    minTime.append(find.min())
    maxTime.append(find.max())
    avgTime.append(find.mean())
    
nSize = [2 ** i for i in range(7, 12)]

plt.plot(nSize, minTime, marker='o', label='Min Time')
plt.plot(nSize, maxTime, marker='s', label='Max Time')
plt.plot(nSize, avgTime, marker='^', label='Average Time')
plt.xlabel("n/2^6")
plt.ylabel("ratio")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig("./ratio plot3.jpg")

plt.clf()

def countOnes(A, n):
    for i in range(n):
        pass