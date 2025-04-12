import nbformat
from nbformat.v4 import new_notebook, new_code_cell
# 전체 코드를 다시 읽어서, 처음부터 끝까지 모든 줄을 셀로 분리하여 포함

# 사용자 코드 불러오기 (완전한 버전으로)
full_code = '''
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from bisect import bisect_left

#======================
# 전역 Random Seed 설정
np.random.seed(20240454)
random.seed(0)

#======================
# 1. Lists

# 1-1. 리스트 복사: assign 함수는 깊은 복사를 통해 원래 리스트와 독립된 복사본을 반환함.
def assign(La):
    return La.copy()

# Algorithm 2: assign 함수 테스트
La = [1]
Lb = assign(La)
La[0] = 0
print("ret of assign function, Lb[0] =", Lb[0])  # 기대: 1

# 1-2. 삭제 시간 측정
# [수정] 리스트 삭제는 의도에 따라 첫 요소를 pop하도록 하여(즉, li.pop(0)) 리스트 삭제 속도가 훨씬 느리게 함.
def listDeletionTime(n):
    times = np.zeros(1000)
    for i in range(1000):
        li = list(range(n))
        start = time.perf_counter()
        # 첫 요소부터 하나씩 삭제하므로 매번 재정렬이 이루어져 시간복잡도가 O(n^2)가 됨.
        while li:
            li.pop(0)
        times[i] = time.perf_counter() - start
    return times

def arrayDeletionTime(n):
    times = np.zeros(1000)
    for i in range(1000):
        arr = np.arange(n)
        start = time.perf_counter()
        # np.delete는 내부에서 C로 최적화되어 전체 배열을 한 번에 삭제하므로 빠름.
        arr = np.delete(arr, np.arange(len(arr)))
        times[i] = time.perf_counter() - start
    return times

# n=100인 경우 각각의 히스토그램 저장
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

# n = 100, 300, 500, 700, 900에서 삭제 시간 비율(리스트/배열) 측정 및 출력, 플롯 저장
ns = [100, 300, 500, 700, 900]
deletion_ratios = []
print("\n[Deletion Time Ratios (list / array)]:")
for n in ns:
    listT = listDeletionTime(n)
    arrayT = arrayDeletionTime(n)
    ratio = listT / arrayT
    deletion_ratios.append([ratio.min(), ratio.mean(), ratio.max()])
    print("n={}: min={:.6e}, avg={:.6e}, max={:.6e}".format(n, ratio.min(), ratio.mean(), ratio.max()))
deletion_ratios = np.array(deletion_ratios)
plt.plot(ns, deletion_ratios[:,0], label="min")
plt.plot(ns, deletion_ratios[:,1], label="avg")
plt.plot(ns, deletion_ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./ListVsArrayDel")
plt.close()

# 1-3. 연결(Concatenation) 시간 측정
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

# n=100인 경우 각각 히스토그램 저장
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

# n = 100,300,500,700,900에서 연결 시간 비율(리스트/배열) 측정 및 출력, 플롯 저장
concatenation_ratios = []
print("\n[Concatenation Time Ratios (list / array)]:")
for n in ns:
    listT = listConcatTime(n)
    arrayT = arrayConcatTime(n)
    ratio = listT / arrayT
    concatenation_ratios.append([ratio.min(), ratio.mean(), ratio.max()])
    print("n={}: min={:.6e}, avg={:.6e}, max={:.6e}".format(n, ratio.min(), ratio.mean(), ratio.max()))
concatenation_ratios = np.array(concatenation_ratios)
plt.plot(ns, concatenation_ratios[:,0], label="min")
plt.plot(ns, concatenation_ratios[:,1], label="avg")
plt.plot(ns, concatenation_ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./ListVsArrayCon")
plt.close()

#======================
# 2. List Extensions (Polynomials)

# 2-1. (a) PolyList: 다항식 f(x)= a_n x^n + ... + a_0 의 리스트 표현에서 0이 아닌 항만 [지수, 계수] 형태로 반환
def PolyList(f):
    deg = len(f) - 1
    return [[deg - i, c] for i, c in enumerate(f) if c != 0]

# g(x)= 3x^8 + 2x  → 표현: [3] + [0]*6 + [2,0]
g = [3, 0, 0, 0, 0, 0, 0, 2, 0]
print("\nPolyList(g) for g(x)=3x^8+2x:", PolyList(g))

# 2-1. (b) PolyEvalList, PolyEvalArray: f(x)= x^50 - x, c = 1/2
def PolyEvalList(f, c):
    t = PolyList(f)
    ret = 0
    for exp, coef in t:
        ret += coef * (c ** exp)
    return ret

def PolyEvalArray(f, c):
    return np.polyval(f, c)

# f(x)= x^50 - x : 계수 표현은 최고차항부터 → [1] + [0]*48 + [-1, 0] (길이 51, 최고차 50)
fx = [1] + [0]*48 + [-1, 0]
ratio = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyEvalList(fx, 0.5)
    t1 = time.perf_counter() - start
    
    start = time.perf_counter()
    PolyEvalArray(fx, 0.5)
    t2 = time.perf_counter() - start
    
    ratio[i] = t1 / t2

plt.hist(ratio, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyEval")
plt.close()

# 2-1. (c) PolyAddList, PolyAddArray:  
# f(x)= x^30 - x + 1 → 표현: [1] + [0]*28 + [-1, 1] (길이 31, 최고차 30)
# g(x)= 3x^8 + 2x   → 표현: [3] + [0]*5 + [2, 0]
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
ratio = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyAddList(fx, gx)
    t1 = time.perf_counter() - start
    
    start = time.perf_counter()
    PolyAddArray(np.array(fx), np.array(gx))
    t2 = time.perf_counter() - start
    
    ratio[i] = t1 / t2

plt.hist(ratio)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyAdd")
plt.close()

# 2-1. (d) PolyProdList, PolyProdArray:
# f(x)= x^4 - x^3 - x + 1 → 표현: [1, -1, 0, -1, 1]
# g(x)= 3x^5 - 4x^2 + 2x  → 표현: [3, 0, 0, -4, 2, 0]
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
ratio = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyProdList(fx, gx)
    t1 = time.perf_counter() - start
    
    start = time.perf_counter()
    PolyProdArray(np.array(fx), np.array(gx))
    t2 = time.perf_counter() - start
    ratio[i] = t1 / t2
    
plt.hist(ratio, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyProd")
plt.close()

# 2-1. (e) PolyIntList, PolyIntArray:
# f(x)= x^3 - x^2 - x → 표현: [1, -1, -1, 0] (길이 4, 최고차 3)
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
ratio = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyIntList(fx, c)
    t1 = time.perf_counter() - start
    
    start = time.perf_counter()
    PolyIntArray(fx, c)
    t2 = time.perf_counter() - start
    
    ratio[i] = t1 / t2
    
plt.hist(ratio, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyInt")
plt.close()

# 2-1. (f) PolyDiffList, PolyDiffArray:
# f(x)= 4x^10 + 2x^7 + 6x^6 - x^4 - 1 → 표현: [4, 0, 0, 2, 6, 0, -1, 0, 0, 0, -1] (길이 11, 최고차 10)
def PolyDiffList(f):
    n = len(f)
    return [f[i] * (n - i - 1) for i in range(n - 1)]

def PolyDiffArray(f):
    n = len(f)
    deg = np.arange(n - 1, 0, -1)
    return f[:-1] * deg

fx = [4] + [0]*2 + [2, 6] + [0]*1 + [-1] + [0]*3 + [-1]
ratio = np.zeros(1000)
for i in range(1000):
    start = time.perf_counter()
    PolyDiffList(fx)
    t1 = time.perf_counter() - start
    
    start = time.perf_counter()
    PolyDiffArray(np.array(fx))
    t2 = time.perf_counter() - start
    
    ratio[i] = t1 / t2
plt.hist(ratio, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistPolyDiff")
plt.close()

#======================
# 2.2. Sharing
# Algorithm 3: 공유 행렬 A 생성 후, 각 그룹별로 공유하는 요소(행의 인덱스)를 리스트로 변환
def SharingList(A):
    return [list(np.where(A[:, i])[0]) for i in range(A.shape[1])]

A_sharing = np.random.binomial(1, 0.2, size=(20, 10))
sharing_list_ret = SharingList(A_sharing)
print("\nSharingList(A):", sharing_list_ret)

def FindPopularList(A_list):
    from collections import Counter
    flat = [item for sublist in A_list for item in sublist]
    return Counter(flat).most_common(1)[0][0] if flat else None

def FindPopularArray(A):
    return int(np.argmax(np.sum(A, axis=1)))

ratios_sharing = np.zeros(1000)
for i in range(1000):
    A_temp = np.random.binomial(1, 0.2, size=(20, 10))
    A_list_temp = SharingList(A_temp)
    start = time.perf_counter()
    FindPopularList(A_list_temp)
    t1 = time.perf_counter() - start
    start = time.perf_counter()
    FindPopularArray(A_temp)
    t2 = time.perf_counter() - start
    ratios_sharing[i] = t1/t2 if t2 != 0 else 0
plt.hist(ratios_sharing, bins=100)
plt.xlabel("Ratio")
plt.ylabel("Freq")
plt.savefig("./HistSharing")
plt.close()

#======================
# 3. Sets

# 3. (a) Ordered set B 생성 함수 (크기 n, 0~100 정수에서 비복원 추출)
def generateOrderedSet(n):
    return sorted(random.sample(range(0, 101), n))

# 3-1. Subset: 주어진 A가 B의 부분집합인지 확인하는 두 방식
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
    return all(B[bisect_left(B, a)] == a if bisect_left(B, a) < len(B) else False for a in A)

sizes = [10, 30, 50, 70, 90]
subset_ratios = []
print("\n[Subset Time Ratios (subset / subsetFast)] for A = {0,9}:")
for n in sizes:
    temp = []
    for _ in range(1000):
        B_set = generateOrderedSet(n)
        start = time.perf_counter()
        subset([0, 9], B_set)
        t1 = time.perf_counter() - start
        start = time.perf_counter()
        subsetFast([0, 9], B_set)
        t2 = time.perf_counter() - start
        temp.append(t1/t2 if t2 != 0 else 0)
    temp = np.array(temp)
    subset_ratios.append([temp.min(), temp.mean(), temp.max()])
    print("n={}: min={:.6e}, avg={:.6e}, max={:.6e}".format(n, temp.min(), temp.mean(), temp.max()))
subset_ratios = np.array(subset_ratios)
plt.plot(sizes, subset_ratios[:,0], label="min")
plt.plot(sizes, subset_ratios[:,1], label="avg")
plt.plot(sizes, subset_ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./Subset")
plt.close()

# 3-2. (b) Union: 두 방식 (non-destructive vs destructive)
# A = {0, 1, 4, 5, 8, 9, 10}
A_union = [0, 1, 4, 5, 8, 9, 10]

def UnionNon(A, B):
    # 두 정렬된 집합 A, B의 합집합 (비파괴적): merge 알고리즘
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
    # 파괴적(un-destructive)이 아닌 방식: A 리스트를 직접 수정하여 합집합을 구함.
    for b in B:
        pos = bisect_left(A, b)
        if pos >= len(A) or A[pos] != b:
            A.insert(pos, b)
    return A

union_ratios = []
print("\n[Union Time Ratios (UnionNon / Union)] for A = {0,1,4,5,8,9,10}:")
for n in sizes:
    temp = []
    for _ in range(1000):
        B_union = generateOrderedSet(n)
        start = time.perf_counter()
        _ = UnionNon(A_union, B_union)
        t1 = time.perf_counter() - start
        A_copy = A_union.copy()
        start = time.perf_counter()
        _ = Union(A_copy, B_union)
        t2 = time.perf_counter() - start
        temp.append(t1/t2 if t2 != 0 else 0)
    temp = np.array(temp)
    union_ratios.append([temp.min(), temp.mean(), temp.max()])
    print("n={}: min={:.6e}, avg={:.6e}, max={:.6e}".format(n, temp.min(), temp.mean(), temp.max()))
union_ratios = np.array(union_ratios)
plt.plot(sizes, union_ratios[:,0], label="min")
plt.plot(sizes, union_ratios[:,1], label="avg")
plt.plot(sizes, union_ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./Union")
plt.close()

# 3-3. (c) Intersect: 두 방식 (non-destructive vs destructive)
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
    # A를 직접 수정: A에 속한 원소 중 B에도 없는 원소 제거
    i = 0
    while i < len(A):
        pos = bisect_left(B, A[i])
        if pos >= len(B) or B[pos] != A[i]:
            A.pop(i)
        else:
            i += 1
    return A

intersect_ratios = []
print("\n[Intersection Time Ratios (IntersectNon / Intersect)] for A = {0,1,4,5,8,9,10}:")
for n in sizes:
    temp = []
    for _ in range(1000):
        B_intersect = generateOrderedSet(n)
        start = time.perf_counter()
        _ = IntersectNon(A_union, B_intersect)
        t1 = time.perf_counter() - start
        A_copy = A_union.copy()
        start = time.perf_counter()
        _ = Intersect(A_copy, B_intersect)
        t2 = time.perf_counter() - start
        temp.append(t1/t2 if t2 != 0 else 0)
    temp = np.array(temp)
    intersect_ratios.append([temp.min(), temp.mean(), temp.max()])
    print("n={}: min={:.6e}, avg={:.6e}, max={:.6e}".format(n, temp.min(), temp.mean(), temp.max()))
intersect_ratios = np.array(intersect_ratios)
plt.plot(sizes, intersect_ratios[:,0], label="min")
plt.plot(sizes, intersect_ratios[:,1], label="avg")
plt.plot(sizes, intersect_ratios[:,2], label="max")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("./Intersect")
plt.close()
'''

# 셀을 나누는 기준은 주석이며, 누락 없이 모두 분리
split_lines = []
buffer = []
for line in full_code.split('\n'):
    if line.strip().startswith("#") and not line.strip().startswith("#="):
        if buffer:
            split_lines.append('\n'.join(buffer).strip())
            buffer = []
    buffer.append(line)
if buffer:
    split_lines.append('\n'.join(buffer).strip())

# 노트북 생성
nb = new_notebook()
nb['cells'] = [new_code_cell(cell) for cell in split_lines]

# 저장
output_path = "Full_Assignment_Executed_Split.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

output_path