import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(20240454)

#==============1-1==============
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
slow = np.zeros(1000)
optimized = np.zeros(1000)
for i in range(1000):
    X = np.random.uniform(0, 1, n)
    start = time.perf_counter()
    prefixAverages1(X, n)
    slow[i] = time.perf_counter() - start

    start = time.perf_counter()
    prefixAverages2(X, n)
    optimized[i] = time.perf_counter() - start

plt.hist(slow / optimized, color='red', label='slow / optimized', bins=100)
plt.legend()
plt.xlabel("Time Ratio")
plt.ylabel("Freq")
plt.xscale("log", base=2)
# plt.yscale("log")
plt.savefig("./ratio_hist.jpg")

plt.clf()

minTime = list()
maxTime = list()
avgTime = list()
for i in range(4, 9):
    n = 2 ** i
    slow = np.zeros(1000)
    optimized = np.zeros(1000)
    for j in range(1000):
        X = np.random.uniform(0, 1, n)
        
        start = time.perf_counter()
        prefixAverages1(X, n)
        slow[j] = time.perf_counter() - start
        
        start = time.perf_counter()
        prefixAverages2(X, n)
        optimized[j] = time.perf_counter() - start
    minTime.append((slow / optimized).min())
    maxTime.append((slow / optimized).max())
    avgTime.append((slow / optimized).mean())
    
nSize = [2 ** i for i in range(4, 9)]

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

n = 2 ** 5
k = 2 ** 4
slow = np.zeros(1000)
optimized = np.zeros(1000)
for i in range(1000):
    X = np.random.uniform(0, 1, n)
    start = time.perf_counter()
    movingAverages1(X, n, k)
    slow[i] = time.perf_counter() - start

    start = time.perf_counter()
    movingAverages2(X, n, k)
    optimized[i] = time.perf_counter() - start

plt.hist(slow / optimized, color='red', label='slow / optimized', bins=100)
plt.legend()
plt.xlabel("Time Ratio")
plt.ylabel("Freq")
plt.xscale("log", base=2)
# plt.yscale("log")
plt.savefig("./ratio_hist2.jpg")

plt.clf()

minTime = list()
maxTime = list()
avgTime = list()
k = 2 ** 3
for i in range(4, 9):
    n = 2 ** i
    slow = np.zeros(1000)
    optimized = np.zeros(1000)
    for j in range(1000):
        X = np.random.uniform(0, 1, n)
        start = time.perf_counter()
        movingAverages1(X, n, k)
        slow[j] = time.perf_counter() - start
        
        start = time.perf_counter()
        movingAverages2(X, n, k)
        optimized[j] = time.perf_counter() - start
    minTime.append((slow / optimized).min())
    maxTime.append((slow / optimized).max())
    avgTime.append((slow / optimized).mean())
    
nSize = [2 ** i for i in range(4, 9)]

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

#==============1-3==============
def findMissing(A, n):
    return (n * (n - 1) // 2) - sum(A)

six = np.zeros(1000)
n = 2 ** 6
for i in range(1000):
    A = list(range(n))
    np.random.shuffle(A)
    A.pop()
    start = time.perf_counter()
    findMissing(A, n)
    six[i] = time.perf_counter() - start

seven = np.zeros(1000)
n = 2 ** 7
for i in range(1000):
    A = list(range(n))
    np.random.shuffle(A)
    A.pop()
    start = time.perf_counter()
    findMissing(A, n)
    seven[i] = time.perf_counter() - start

plt.hist(seven / six, color='red', label='2^7/2^6', bins=100)
plt.legend()
plt.xlabel("Time Ratio")
plt.ylabel("Freq")
plt.xscale("log", base=2)
# plt.yscale("log")

plt.savefig("./ratio_hist3.jpg")
plt.clf()

three = np.zeros(1000)
n = 2 ** 3
for i in range(1000):
    A = list(range(n))
    np.random.shuffle(A)
    A.pop()
    start = time.perf_counter()
    findMissing(A, n)
    three[i] = time.perf_counter() - start

minTime = list()
maxTime = list()
avgTime = list()
for i in range(4, 9):
    n = 2 ** i
    find = np.zeros(1000)
    for j in range(1000):
        X = np.random.uniform(0, 1, n)
        start = time.perf_counter()
        findMissing(X, n)
        find[j] = time.perf_counter() - start
    minTime.append((find / three).min())
    maxTime.append((find / three).max())
    avgTime.append((find / three).mean())
    
nSize = [2 ** i / 2 ** 3 for i in range(4, 9)]

plt.plot(nSize, minTime, marker='o', label='Min Time')
plt.plot(nSize, maxTime, marker='s', label='Max Time')
plt.plot(nSize, avgTime, marker='^', label='Average Time')
plt.xlabel("2^n/2^3")
plt.ylabel("ratio")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig("./ratio plot3.jpg")

plt.clf()

#==============1-4==============
def countOnes(A, n):
    loc = n - 1
    cnt = 0
    for i in range(n):
        while A[i][loc] == 0:
            if loc == 0:
                return cnt
            loc -= 1
        cnt += loc + 1
    return cnt

def countOnesButSlow(A, n):
    c = 0
    for i in range(n):
        j = 0
        while j < n and A[i][j] == 1:
            c += 1
            j += 1
    return c

n = 2 ** 6
slow = np.zeros(1000)
optimized = np.zeros(1000)
for i in range(1000):
    A = np.zeros((n, n))
    numOfOne = sorted(np.random.randint(0, n + 1, size=n), reverse=True)
    for j in range(0, n):
        A[j, :numOfOne[j]] = 1
    
    start = time.perf_counter()
    countOnesButSlow(A, n)
    slow[i] = time.perf_counter() - start

    start = time.perf_counter()
    countOnes(A, n)
    optimized[i] = time.perf_counter() - start
    
plt.hist(slow / optimized, color='red', label='countOnesButSlow / countOnes', bins=100)
plt.legend()
plt.xlabel("Time Ratio")
plt.ylabel("Freq")
plt.xscale("log", base=2)
# plt.yscale("log")

plt.savefig("./ratio_hist4.jpg")

plt.clf()

minTime = list()
maxTime = list()
avgTime = list()
for i in range(4, 9):
    n = 2 ** i
    slow = np.zeros(1000)
    optimized = np.zeros(1000)
    for j in range(1000):
        A = np.zeros((n, n))
        numOfOne = sorted(np.random.randint(0, n + 1, size=n), reverse=True)
        for k in range(0, n):
            A[k, :numOfOne[k]] = 1
        
        start = time.perf_counter()
        countOnesButSlow(A, n)
        slow[j] = time.perf_counter() - start

        start = time.perf_counter()
        countOnes(A, n)
        optimized[j] = time.perf_counter() - start
    minTime.append((slow / optimized).min())
    maxTime.append((slow / optimized).max())
    avgTime.append((slow / optimized).mean())
    
nSize = [2 ** i for i in range(4, 9)]

plt.plot(nSize, minTime, marker='o', label='Min Time')
plt.plot(nSize, maxTime, marker='s', label='Max Time')
plt.plot(nSize, avgTime, marker='^', label='Average Time')
plt.xlabel("n")
plt.ylabel("ratio")
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig("./ratio plot4.jpg")

plt.clf()

#==============2-1===============
def gcd1(a, b):
    if b == 0:
        return a
    
    print(f"computing gcd1({a}, {b}), ", end='')
    
    return gcd1(b, a % b)

def gcd2(a, b):
    if a == b:
        return a
    
    print(f"computing gcd2({a}, {b}), ", end='')
    
    if a > b:
        return gcd2(a - b, b)
    else:
        return gcd2(a, b - a)

print(f"and gcd1(493, 33) is {gcd1(493, 33)}")
print()
print(f"and gcd2(493, 33) is {gcd2(493, 33)}")
print()
print(f"and gcd1(225, 13) is {gcd1(225, 13)}")
print()
print(f"and gcd2(225, 13) is {gcd2(225, 13)}")
print()

#==============2-2===============
def divide(a, b):
    if a < b:
        return (0, a)
    
    print(f"computing divide({a}, {b}), ", end='')
    
    q, r = divide(a - b, b)
    return (q + 1, r)

q, r = divide(413, 31)
print(f"and divide(413, 31) is quotient: {q}, remainder: {r}")
print()
q, r = divide(1325, 113)
print(f"and divide(1325, 113) is quotient: {q}, remainder: {r}")
print()

#==============3-1===============
def spiral(n, m):
    A = np.zeros((n, m), dtype=int)
    x, y = 0, 0
    togo = ((1, 0), (0, 1), (-1, 0), (0, -1))
    now = 0
    for i in range(1, n * m + 1):
        A[y][x] = i
        nx, ny = x + togo[now][0], y + togo[now][1]
        if m <= nx or n <= ny or A[ny][nx] != 0:
            now = (now + 1) % 4
        x += togo[now][0]
        y += togo[now][1]
    return A

print(spiral(10, 4))

#==============3-2===============
n = 10 ** 6
toHist = np.zeros(n)
for i in range(n):
    A = np.zeros((10, 10))
    howMuch = np.random.randint(1, 6)
    loc = np.random.choice(range(100), howMuch, replace=False)
    A.flat[loc] = 1
    nonzeroOffset = np.sum(loc)
    locOffset = ((howMuch * 2) * (howMuch * 2 - 1)) // 2
    toHist[i] = nonzeroOffset / locOffset

plt.hist(toHist, color='red', label='offset ratio', bins=100)
plt.legend()
plt.xlabel("Offset Ratio")
plt.ylabel("Freq")
plt.xscale("log", base=2)

plt.savefig("./sparse hist1.jpg")

plt.clf()

n = 10 ** 6
toHist = np.zeros(n)
for i in range(n):
    A = np.zeros((10, 10))
    howMuch = np.random.randint(10, 21)
    loc = np.random.choice(range(100), howMuch, replace=False)
    A.flat[loc] = 1
    nonzeroOffset = np.sum(loc)
    locOffset = ((howMuch * 2) * (howMuch * 2 - 1)) // 2
    toHist[i] = nonzeroOffset / locOffset

plt.hist(toHist, color='red', label='offset ratio', bins=100)
plt.legend()
plt.xlabel("Offset Ratio")
plt.ylabel("Freq")
# plt.xscale("log", base=2)

plt.savefig("./sparse hist2.jpg")

plt.clf()