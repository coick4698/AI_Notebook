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

# 선택 정렬
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