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
    - Pivot(기준값)을 선택한 뒤, 작은 값과 큰 값을 분리하여 재귀적으로 정렬하는 알고리즘
    *** Pivot을 기준으로 분할하고 양쪽을 각각 정렬한다***
    - Divide and Conquer를 사용한다.
    
    ex)
    arr = [5,2,8,1,7,3]
    Pivot = 5
    Pivot보다 작은 값 = [2,1,3]
    Pivot보다 큰 값 = [8,7]
    분할 결과: [2,1,3]+[5]+[8,7]
    분할된 각 배열에 다시 재귀적으로 퀵 정렬 적용
    quick_sort([2,1,3]) --> [1,2,3]
    quick_sort([8,7]) --> [7,8]
    최종: [1,2,3,5,7,8]
    
    Time Complexity:
    - Best    : O(N log N) # Pivot이 매번 배열을 절반씩 나눌 경우(logN) * 각 단계의 파티션 작업(N)
    - Average : O(N log N) # Pivot을 매번 최대/최솟값으로 선택할 경우(N) * 각 단계의 파티션 작업(N)
    - Worst   : O(N^2)
    
    퀵 정렬을 떠올릴 수 있는 상황:
    - Pivot을 기준으로 데이터를 분할하는 문제
    - Partition 개념을 활용하는 문제
    - K번째 작은 값을 찾는 문제
"""

def quick_sort(arr):
    n = len(arr) # n은 arr의 길이
    
    if len(arr) <= 1: # Base Case: arr의 원소가 하나라면 즉시 반환
        return arr
    
    pivot = arr[0] # 피봇은 arr의 첫번째 요소
    
    # Partition
    left = [] # 피봇보다 작은 값들이 들어갈 리스트
    right = [] # 피봇보다 큰 값들이 들어갈 리스트
    
    for x in arr[1:]: # 피봇 다음 요소부터 리스트 끝까지 순환하면서 체크
        if x <= pivot: # 피봇보다 작은 값이면
            left.append(x) # left 리스트에 삽입
        
        else: # 피봇보다 큰 값이면
            right.append(x) # right 리스트에 삽입
    
    return quick_sort(left) + [pivot] + quick_sort(right) # 분할된 리스트들을 대상으로 재귀적으로 퀵 정렬을 실행하여 합친 후 반환
# Heap Sort(힙 정렬) -------------------------------------------------------------
"""
    - 완전이진 트리 구조로 최댓값 또는 최솟값을 반복적으로 추출하는 정렬 알고리즘
    ***최대 힙을 만들고 최댓값을 배열 뒤쪽에 배치한다***
    
    [Heap의 특징]
    - Complete Binary Tree
    - 부모와 자식 사이의 대소 관계를 유지한다
    - 전체 배열이 정렬되어 있는 것은 아니다
    
    [Heap의 종류]
    1. Max Heap
    - 부모 노드 >= 자식 노드
    - 루트 노드는 전체 최댓값
    2. Min Heap
    - 부모 노드 <= 자식노드
    - 루트 노드는 전체 최솟값
    
    ex)
    arr = [4,10,3,5,1]
    Max Heap: [10,5,3,4,1]
    Heap Sort:
    [10,5,3,4,1] --> 루트 노드인 10을 마지막 1과 교환
    [1,5,3,4 | 10] --> 10 정렬 완료
    정렬되지 않은 영역을 Heapify:
    [5,4,3,1 | 10] --> 루트 노드인 5를 마지막 1과 교환
    [1,4,3 | 5,10] --> 5 정렬 완료
    정렬되지 않은 영역을 Heapify: 
    [4,3,1 | 5,10] --> 루트 노드인 4를 마지막 1과 교환
    [1,3 | 4,5,10] --> 4 정렬 완료
    정렬되지 않은 영역을 Heapify:
    [3,1 | 4,5,10] --> 루트 노드인 3을 마지막 1과 교환
    [1,3,4,5,10] --> 정렬완료
    
    Time Complexity:
    - Best    : O(N log N)
    - Average : O(N log N)
    - Worst   : O(N log N)
    
    힙 정렬을 떠올릴 수 있는 상황:
    - 최악 O(N log N) 정렬이 필요한 경우
    - 추가 메모리를 적게 사용하는 정렬 문제
    - K번째로 큰 값 구하기
    - 우선순위가 가장 높은 작업 처리
    - 가장 작은/큰 값 반복 추출
    - 다익스트라 알고리즘
    
"""

def heapify(arr,n,i): # Heap의 조건
    # n은 Heap으로 취급할 영역의 크기 // i는 heapify를 시작할 부모 노드 index
    largest = i # 현재 노드를 가장 큰 값으로 가정
    
    left = 2*i+1 # 왼쪽 자식의 인덱스
    right = 2*i+2 # 오른쪽 자식의 인덱스
    
    if left < n and arr[left] > arr[largest]: # 현재 노드를 왼쪽 자식과 비교
        largest = left
    
    if right < n and arr[right] > arr[largest]: # 현재 노드를 오른쪽 자식과 비교
        largest = right
        
    if largest != i: # 부모가 최댓값이 아니라면
        arr[i], arr[largest] = arr[largest], arr[i] # Swap
        
        heapify(arr,n,largest)# 영향받은 하위 트리의 Heap 조건 복구
        
def heap_sort(arr):
    n = len(arr) # n은 arr의 전체 길이
    
    # Step 1: Max Heap 만들기
    for i in range(n//2-1,-1,-1):
        heapify(arr,n,i)
    
    # Step 2: Extract Maximum
    for i in range(n-1, 0 , -1):
        
        arr[0], arr[i] = arr[i], arr[0] # 최댓값을 끝으로 이동(정렬완료)
        heapify(arr,i,0) # 남은 Heap 복구
    
    return arr

# Counting Sort(계수 정렬) -------------------------------------------------------
"""
    - 각 숫자가 몇 번 등장했는지 계산한 다음, 그 개수만큼 순서대로 배치하는 방식
    ***각 숫자의 개수를 세고, 그 개수만큼 출력한다***
    - 원소끼리 크기를 비교 X
    - 정수 데이터에 적합
    - 숫자의 범위가 작을 때 효율적
    - 빈도 계산에 적합
    
    ex)
    arr = [3,1,2,3,2,1,3]
    Step 1: Counting
    숫자 0 -> 0개
    숫자 1 -> 2개
    숫자 2 -> 2개
    숫자 3 -> 3개
    count = [0,2,2,3]
    Step 2: Reconstruct
    최종 어레이에
    숫자 0을 0번 추가
    숫자 1을 2번 추가
    숫자 2를 2번 추가
    숫자 3을 3번 추가
    최종: [1,1,2,2,3,3,3]

"""
def counting_sort(arr):
    if not arr: # 만약 비어있는 배열이라면
        return [] # 그대로 반환
    
    max_value = max(arr) # 현재 배열의 최댓값 확인
    
    count = [0] * (max_value + 1) # 빈도수를 저장할 배열 생성
    
    for num in arr: # arr에서 각 숫자마다 빈도 수 체크
        count[num] += 1
    
    result = [] # 빈도 수로 정렬된 결과 arr
    
    for num in range(len(count)):
        
        for  _ in range(count[num]):
            result.append(num)
    
    return result