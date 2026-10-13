"""
    문제 유형: 정렬
    
    대표 알고리즘/자료구조:
    - 선택 정렬
    - 버블 정렬
    - 삽입 정렬
    - 병합 정렬
    - 퀵 정렬
    - 힙 정렬
    - 계수 정렬
    
    특징:
    - 가장 큰/작은
    - 순서대로
    - 순위
    - N번째로 큰 값
    - 오름차순/내림차순
    - 두 배열을 비교
    - 정렬 후 선택
"""

# Selection Sort(선택 정렬) ------------------------------------------------------
"""
    - 현재 위치에 들어갈 가장 작은 값을 남아 있는 데이터에서 찾아 교환하는 정렬 알고리즘
    *** "남은 범위에서 가장 작은 값을 찾아 앞으로 보낸다." ***
    
    ex)
    arr =  [5, 2, 8, 1, 3]
    최솟값 = 1
    첫번째 자리의 5와 최솟값인 1의 위치를 교환
    arr = [1, 2, 8, 5, 3]
    arr = [1, | 2, 8, 5, 3]
    인덱스 0 과 1사이에 벽이 있다고 생각하고 인덱스 1부터 같은 프로세스 진행
    
    Time Complexity:
    - Best Case    : O(N^2)
    - Average Case : O(N^2)
    - Worst Case   : O(N^2)
    
    선택 정렬을 떠올릴 수 있는 상황:
    - 남은 데이터에서 최솟값/최댓값을 반복적으로 선택하는 구조
    - 거의 arr.sort()나 sorted(arr)로 대체 가능
    
"""
def selection_sort(arr):
    n = len(arr) # arr의 길이
    
    for i in range(n):
        min_idx = i # 현재 위치 i를 최솟값이라고 가정
        
        for j in range(i+1, n): # 아직 정렬되지 않은 벽 오른쪽 영역 탐색
            
            if arr[j] < arr[min_idx]: # 더 작은 값을 발견한다면
                min_idx = j # 그 값을 index로 갱신
            
        arr[i],arr[min_idx] = arr[min_idx], arr[i] # 현재 위치와 실제 최솟값을 교환
    
    return arr # 정렬된 arr 반환

# Bubble Sort(버블 정렬) ---------------------------------------------------------
"""
    - 인접한 두 원소를 비교하고 필요할 때마다 교환(스왑)하는 방식
    *** 인접한 두 값을 비교하며 큰 값을 오른쪽으로 보낸다 / 큰 수는 거품처럼 뒤로 떠오른다 ***
    
    ex)
    arr = [5, 2, 8, 1, 3]
    1트:
    [5, 2, 8, 1, 3] -- 5 > 2 so swap
    [2, 5, 8, 1, 3] -- 5 < 8 so stay
    [2, 5, 8, 1, 3] -- 8 > 1 so swap
    [2, 5, 1, 8, 3] -- 8 > 3 so swap finish
    [2, 5, 1, 3, | 8]
    2트:
    arr = [2, 5, 1, 3, | 8]
    [2, 5, 1, 3, | 8] -- 5 > 1 so swap
    [2, 1, 5, 3, | 8] -- 5 > 3 so swap
    [2, 1, 3, | 5, 8] -- 5 < 8 so stay finish
    3트:
    arr = [2, 1, 3, | 5, 8]
    [2, 1, 3, | 5, 8] -- 2 > 1 -- so swap
    [1, 2, | 3, 5, 8] -- 2 < 3 -- so stay finish
    
    Time Complexity:
    - Best    : O(N^2)
    - Average : O(N^2)
    - Worst   : O(N^2)
    
    버블 정렬을 떠올릴 수 있는 상황:
    - 인접한 두 원소를 비교하여 정렬하는 문제
    - 인접한 원소끼리만 swap할 수 있는 문제
    - 정렬 과정에서 swap 횟수를 분석하는 문제
"""

def bubble_sort(arr):
    n = len(arr) # arr의 길이
    
    for i in range(n-1): # 1개 정해 놓고 반복돌리니까 (전체길이 - 1)만큼 반복
        
        for j in range(n-1-i): # j+1에 접근하므로 마지막 index 넘는걸 방지
            if arr[j] > arr[j+1]: # 원소 비교해서 왼쪽이 더 크면
                arr[j],arr[j+1] = arr[j+1],arr[j] # 오른쪽 원소랑 자리 교체
    
    return arr # 정렬된 arr 반환


# Insertion Sort(삽입 정렬) ------------------------------------------------------
"""
    - 이미 정렬된 영역에서 새로운 원소가 들어갈 위치를 찾아 삽입하는 방식
    *** 왼쪽의 정렬된 영역에 새로운 원소를 적절한 위치로 삽입 ***
    
    ex)
    arr = [5, 2, 8, 1, 3]
    초기 상태:
    [5 | 2, 8, 1, 3] --> 5는 이미 정렬되었다고 가정
    Pass: 1 key = 2
    [5 | 2, 8, 1, 3] --> 5 > 2 이므로 5를 오른쪽으로 이동
    [2, 5 | 8, 1, 3]
    Pass: 2 key: 8
    [2, 5 | 8, 1, 3] --> 5 < 8 이므로 이동 x
    [2, 5, 8 | 1, 3]
    Pass: 3 key: 1
    [2, 5, 8 | 1, 3] --> 8 > 1 이므로 8을 오른쪽으로 이동
    [1, 2, 5, 8 | 3]
    Pass: 4 key: 3
    [1, 2, 5, 8 | 3] --> 8 > 3 이므로 8과 5를 오른쪽으로 이동, 3을 2와 5 사이 삽입
    [1, 2, 3, 5, 8]
    
    구조: [정렬된 영역 | key | 정렬되지 않은 영역]
    
    Time Complexity:
    - Best    : O(N)
    - Average : O(N^2)
    - Worst   : O(N^2)
    
    삽입 정렬을 떠올릴 수 있는 상황:
    - 정렬된 데이터에 새로운 값을 삽입하는 구조
    - 데이터가 거의 정렬되어 있는 경우
    - 입력 데이터의 크기가 작은 경우
"""

def insertion_sort(arr):
    n = len(arr) # n은 arr의 길이
    
    for i in range(1, n): # 첫번째 원소가 정렬되어 있다고 가정
        key = arr[i] # 정렬된 영역에 삽입할 값
        j = i - 1 # key 바로 왼쪽부터 탐색
        
        while j >= 0 and arr[j] > key: # key보다 큰 원소들을 오른쪽으로 이동
            arr[j+1] = arr[j]
            j -= 1
        
        arr[j+1] = key # key를 올바른 위치에 삽입
    
    return arr # 정렬된 arr 반환

# Merge Sort(병합 정렬) ----------------------------------------------------------
"""
    - 배열을 계속 반으로 나눈 뒤, 정렬된 작은 배열을 병합하여 전체를 정렬하는 알고리즘
    *** 반으로 나누고, 정렬하면서 다시 합친다 ***
    - Divide and Conquer (분할 정복)를 사용
    
    ex)
    arr = [8,3,5,1]
    Divide:
    [8,3],[5,1] --> 반으로 나눔
    [8],[3],[5],[1] --> 원소가 1개인 배열은 이미 정렬된 상태!
    Merge:
    [3,8][1,5] --> 한개짜리 원소들을 비교하여 두개인 리스트로 순서대로 병합시킴
    [1,3,5,8] --> 차례차례 두 리스트의 원소들을 비교하면서 병합시킴
    
    Time Complexity:
    - Best    : O(N log N)
    - Average : O(N log N)
    - Worst   : O(N log N)
    
    병합 정렬을 떠올려야 할 때:
    - O(NlogN) 정렬이 필요한 경우
    - Divide and Conquer를 사용하라는 문제
    - 정렬된 두 배열을 효율적으로 합쳐야할 때
    - Inversion Counting(역전쌍 계산)
"""

def merge_sort(arr):
    n = len(arr) # arr의 길이
    
    if n <= 1: # Base Case(arr의 원소가 하나라면 정렬된 상태니까)
        return arr # 바로 반환
    
    # Divide Part
    mid = n // 2
    left = merge_sort(arr[:mid]) # 재귀
    right = merge_sort(arr[mid:])
    
    # Merge Part
    result = [] # 원소를 하나씩 넣을 빈 리스트
    i,j = 0,0 # left,right 배열의 인덱스
    
    while i < len(left) and j < len(right): # 인덱스 i,j가 각 배열의 끝까지 순환할 때까지
        
        if left[i] <= right[j]: # 만약 왼쪽 배열이 더 작으면
            result.append(left[i]) # 더 작은 값을 결과 리스트에 저장
            i += 1 # left 배열의 다음 원소로 인덱스 이동
            
        else: # 오른쪽 배열의 원소가 더 작다면
            result.append(right[j]) # 오른쪽 배열의 원소를 결과 리스트에 저장
            j += 1 # right 배열의 다음 원소로 인덱스 이동
    
    # 남은 원소를 모두 추가(한쪽의 원소들만 계속 result에 추가됐다면 반대 리스트에는 원소들이 남아있으니까)
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result # 정렬되어 병합된 arr 반환
    
# Quick Sort(퀵 정렬) ------------------------------------------------------------
"""

"""

# Heap Sort(힙 정렬) -------------------------------------------------------------
"""

"""

# Counting Sort(계수 정렬) -------------------------------------------------------
"""

"""
