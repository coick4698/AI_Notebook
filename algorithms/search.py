"""
    문제유형: 탐색
    
    대표 알고리즘/자료구조:
    - Linear Search(선형 탐색)
    - Binary Search(이진 탐색)
    - Brute Force(완전탐색)
    - DFS(깊이 우선 탐색)
    - BFS(넓이 우선 탐색)
    
    특징:
    - 가능한 경우를 찾아라
    - 최소 몇 번 이동?
    
"""
# Linear Search(선형 탐색) -------------------------------------------------------
"""
    ***처음부터 하나씩 비교하며 찾는다***
    - 배열의 첫 번째 원소부터 마지막 원소까지 하나씩 확인하면서 원하는 값을 찾는 알고리즘
    - 정렬되지 않은 배열에서도 사용 가능
    - 구현이 간단
    - 최악의 경우, 모든 원소를 확인 --> 입력 크기 N에 비례해서 탐색 시간이 증가
    
    Time Complexity:
    - Best    : O(1)
    - Average : O(N)
    - Worst   : O(N)
    
    Space Complexity:
    - O(1)
    
    선형 탐색을 떠올릴 수 있는 상황:
    - 배열이 정렬되어 있지 않은 경우
    - 데이터의 크기가 작은 경우
    - 특징 값의 첫 번째 index를 찾는 경우
    - 배열을 한 번 순회하며 조건을 검사하는 경우
"""

def linear_search(arr, target):
    for i in range(len(arr)): # 리스트의 길이만큼 반복(모든 인덱스 참조)
        if arr[i] == target:
            return i # 타겟 반환
    return -1 # 타겟이 리스트에 없으면 -1 반환


# Binary Search(이진 탐색) -------------------------------------------------------
"""
    ***중간값과 비교하여 불필요한 절반을 제거한다***
    - 정렬된 배열에서 탐색 범위를 절반씩 줄여 타겟을 찾는 알고리즘
    - 배열이 정렬되어 있어야한다
    - 탐색 범위를 절반씨 제거할 수 있다
    
    Time Complexity:
    - Best    : O(1)
    - Average : O(log N)
    - Worst   : O(log N)
    
    Space Complexity:
    - O(1)
    
    이진 탐색을 떠올릴 수 있는 상황:
    - 정렬된 배열에서 특정 값을 찾는 경우
    - 데이터 크기가 매우 큰 경우
    - 특정 값의 존재 여부를 빠르게 확인하는 경우
    - 특정 값의 첫 번째/ 마지막 위치를 찾는 경우
    - Lower Bound / Upper Bound 문제
    - 정답의 범위에서 조건을 만족하는 경계값을 찾는 문제(Parameter Search)
"""

def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left+right) // 2 # 전체의 중간값(짝수일 경우 왼쪽 선택)
        
        if arr[mid] == target: # 타겟 발견하면
            return mid # 즉시 반환
        elif arr[mid] < target: # 타겟이 오른쪽에 있다면
            left = mid + 1 # left를 mid 한 칸 오른쪽으로
        else: # 타겟이 왼쪽에 있다면
            right = mid - 1 # right를 mid 한 칸 왼쪽으로

# Brute Force(완전탐색) -------------------------------------------------------------
"""
    ***가능한 모든 경우를 확인해서 정답을 찾는 문제 해결 방식***
    1. Basic Brute Force: 반복문으로 모든 경우 확인
    - 모든 가능한 경우를 확인한다.
    - 구현이 직관적이다.
    - 정답을 빠뜨리지 않는 것이 중요하다.
    - 경우의 수가 많으면 비효율적이다.
    
    Time Complexity:
    - 문제의 따라 다름
    
    Brute Force를 떠올릴 수 있는 상황:
    - 입력 데이터의 크기가 작은 경우
    - 모든 경우를 확인해야 하는 경우
    - 가능한 조합이나 순서를 검사하는 경우
    - 다른 효율적인 알고리즘이 떠오르지 않는 경우
    - 최적화 알고리즘의 정답 검증용 구현
    
    
    
    2. Permutation(순열): 순서를 고려하여 원소 선택
    ***순서가 다르면 서로 다른 경우로 취급***
    - 순서를 고려한다.
    - 같은 위치의 원소를 중복 선택하지 않는다.
    - 재귀를 이용하여 구현할 수 있다.
    - 모든 순열을 생성하면 경우의 수가 빠르게 증가한다.
    - 내장 프로그램: from itertools import permutations
    
    Time Complexity:
    - 원소의 개수가 증가하면 경우의 수가 매우 빠르게 증가: O(N * P(N,R))
    
    Space Complexity:
    - O(N + R * P(N,R))
    
    순열을 떠올릴 수 있는 상황:
    - 원소의 순서가 중요한 경우
    - 가능한 모든 배치를 확인해야 하는 경우
    - 서로 다른 순서의 선택을 검사하는 경우
    - 순서에 따라 결과가 달라지는 문제
    - 모든 방문 순서를 탐색하는 문제
    
    
    
    3. Combination(조합): 순서와 관계없이 원소 선택
    ***순서가 달라도 같은 원소를 선택했다면 같은 경우***
    - 순서를 고려하지 않는다.
    - 같은 원소를 중복 선택하지 않는다.
    - 재귀적 구현
    - from itertools import combinations
    Time Complexity:
    - O(R * C(N,R))
    
    Space Complexity:
    - O(R + R * C(N, R))
    
    조합을 떠올릴 수 있는 상황:
    - 순서가 중요하지 않은 경우
    - 특정 개수의 원소를 선택할 경우
    - 팀이나 그룹을 구성하는 경우
    
    
    
    4. Backtracking: 재귀 탐색과 가지치기
    ***선택 -> 탐색 -> 조건 검사 -> 선택 취소***
    - 현재 선택이 잘못되었다면 남은 경우를 탐색하지 않아도 되는가? 그렇다면 Backtracking 고려
    
    Time Complexity:
    - 문제에 따라 다름
    
    Space Complexity:
    - O(N * 2^N)
    
    백트래킹을 떠올릴 수 있는 상황:
    - 가능한 선택지가 여러 개인 경우
    - 선택을 취소하며 다른 경우를 탐색하는 경우
    - 조건을 만족하지 않는 경로를 제거할 수 있는 겨웅
    - 제약 조건이 있는 탐색 문제
    -
"""
def brute_force(arr, target):
    result = [] # 이 배열에 타겟 저장
    
    for i in range(len(arr)): # 배열의 길이만큼 반복
        for j in range(i+1, len(arr)): # 서로 다른 두 인덱스를 순서에 관계없이 한 번씩 선택할 때 자주 사용
            if arr[i] + arr[j] == target:
                result.append(arr[i], arr[j])
    
    return result



def permutation(arr, r):
    result = []
    visited = [False] * len(arr) # 원소 선택 여부 기록; 배열의 길이만큼 False가 들어있는 배열
    path = [] # 현재 선택한 순열
    
    def dfs():
        # Base case: r개 선택 완료
        if len(path) == r:
            result.append(path.copy())
            return
        # 모든 원소 확인
        for i in range(len(arr)):
            if visited[i]: # 이미 선택한 원소라면 건너뛰기
                continue
            visited[i] = True
            path.append(arr[i])
            # 다음 원소 선택
            dfs() # 재귀적
            # 선택 취소 (Backtracking)
            path.pop()
            visited[i] = False
    dfs()
    
    return result



def combination(arr, r):
    result = []
    path = []
    
    def dfs(start):
        # Base Case: r개 선택 완료
        if len(path) == r:
            result.append(path.copy())
            return
        # start index부터 선택
        for i in range(start, len(arr)):
            path.append(arr[i]) # 원소 선택
            dfs(i + 1) # 다음 index부터 탐색
            path.pop() # 선택 취소
    dfs(0)
    
    return result


def backtracking(arr, target):
    result = []
    path = []
    
    def dfs(start, current_sum):
        # Base Case: 목표 합계 달성
        if current_sum == target:
            result.append(path.compy())
            return
        # Pruning: 목표 합계 초과
        if current_sum > target:
            return
        # 가능한 원소 탐색
        for i in range(start, len(arr)):
            # 원소 선택
            path.append(arr[i])
            # 다음 단계 탐색
            dfs(i + 1, current_sum + arr[i])
            # 선택 취소
            path.pop()
    dfs(0,0)
    
    return result

# DFS(깊이 우선 탐색) -------------------------------------------------------------
"""

"""

# BFS(넓이 우선 탐색) -------------------------------------------------------------
"""

"""
