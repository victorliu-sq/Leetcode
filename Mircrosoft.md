# Microsoft

## HashTable&Array&String

### 49. Group Anagrams

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for s in strs:
            arr = [0] * 26
            for c in s:
                arr[ord(c) - ord('a')] += 1
            key = tuple(arr)
            d[key] += [s]
        return d.values()
        '''
        Data Structure:
            1.  d:
                +: iterate each str --> count of its letters --> tuple --> d[tuple] += [s]
        '''
```



### 54. Spiral Matrix

```python
class Solution:
    def spiralOrder(self, mat: List[List[int]]) -> List[int]:
        m, n = len(mat), len(mat[0])
        r1, r2, c1, c2 = 0, m - 1, 0, n - 1
        res = []
        while True:
            if r1 <= r2 and c1 <= c2:
                for i in range(c1, c2 + 1):
                    res += [mat[r1][i]]
                r1 += 1

                for i in range(r1, r2 + 1):
                    res += [mat[i][c2]]
                c2 -= 1
            else:
                break
            
            if r1 <= r2 and c1 <= c2:
                for i in range(c1, c2 + 1)[::-1]:
                    res += [mat[r2][i]]
                r2 -= 1

                for i in range(r1, r2 + 1)[::-1]:
                    res += [mat[i][c1]]
                c1 += 1
            else:
                break
        return res
        '''
        Explanation:
            1.  we can make 4 iterations in a framework is like:
                1 2 3
                4   6
                7 8 9
            
            2   we can only make 2 (1 is ignored) iterations, if the framework is a line:
                (1) 1 2 3 
                or 
                (2) 3 
                    6
                    9
            
        Data Structure:
            1.  r1:
                r1 + 1: after we iterate mat[r1][c1 ~ c2]
            2.  r2:
                r2 - 1: after we iterate mat[r2][c2 ~ c1]
            3.  c1:
                c1 + 1: after we iterate mat[r2 ~ r1][c1]
            4.  c2:
                c2 - 1: after we iterate mat[r1 ~ r2][c2]
                
        Algorithm:
            while true:
            (1) check r1 <= r2, c1 <= c2:
                iterate left to right, up to down
            (2) check r1 <= r2, c1 <= c2:
                iterate right to left, down to up
        '''
```



### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals += [[10001, 10001]]
        res = []
        intervals.sort()
        s, e = intervals[0]
        for ns, ne in intervals[1:]:
            if s <= ns <= e:
                e = max(e, ne)
            else:
                res += [[s, e]]
                s, e = ns, ne
        return res
        '''
        Data Structure:
            1.  s, e:    
                (1) extend interval: if [ns, ne], s <= ns <= e, extend interval e = max(e, ne)
                (2) new interval: if [ns, ne], e < ns, we try to extend interval[ns, ne]
            
            2.  res:
                +: if e < ns, there is no intersection --> add [s, e] to res
        
        Algorithm:
            1.  add a interval with very big start to intervals
                so we can add last interval to res
                
            2.  sort intervals by (1) start (2) end
            
            3.  iterate each interval:
                (1) if intersection: extend
                
                (2) else: add [s, e] to res
        '''
```



### 151. Reverse Words in a String

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        words = s.split(" ")
        n = len(words)
        reversed_words = []
        # print(words)
        for word in words:
            if word != '':
                reversed_words += [word[::-1]]
                
        return " ".join(reversed_words)[::-1]
                
        
        '''
        Data Structure:
            s -> words -> word
            1. s: reverse
            2. word: reverse
            
        Algorithm:
            (1) reverse word in words
            (2) combine all words together and reverse the whole string
        '''
```



### 233. Number of Digit One

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        x = 1
        res = 0
        
        while x <= n:
            upperT = n // x
            T = upperT % 10
            lower = n % x
            upper = upperT // 10
            
            countUpper = upper * x
            countLower = 0
            
            if T == 1:
                countLower = lower + 1
            elif T > 1:
                countLower = x
            
            res += countUpper + countLower
            x = x * 10
        return res
    """
    Explanation:
        xyz T abc:
        xyz --> upper
        abc --> lower
        (1) 001 T abc --> xyz T abc
            countUpper: xyz * 1000

        (2) <1> T == 1:
                xyz 1 abc --> xyz 1 000
                countLower: abc + 1

            <2> T > 1:
                xyz 1 000 ~ xyz 1 999 
                countLower: 1000

    Data Structure:
        1.  x: 1, 10, 100, 1000...
            (1) * 10: after adding count of 1 in position T
        
        2.  countUpper: 
            upper * x
        
        3.  countLower:
            (1) T = 1: lower + 1
            (2) T > 1: lower
            
    Algorithm
        1.initialze x = 1

        2.  while x <= n:
            (1) calculate upper, T, lower
            (2) calculate countUpper
            (3) calculate countLower
            (4) x *= 10
        """
```



## LinkedList

### 23. Merge k Sorted Lists

```python
from heapq import heappush, heappop
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        h = []
        for i, head in enumerate(lists):
            if head:
                heappush(h, (head.val, i, head))
        
        dummy = ListNode(0)
        cur = dummy
        while h:
            val, idx, node = heappop(h)
            cur.next = node
            cur = cur.next
            
            node = node.next
            if node:
                heappush(h, (node.val, idx, node))
        return dummy.next
        
        '''
        Data Structure:
            1.  h:
                (1) +:
                    <1> initialize: push heads of all linkedlist to h (val, idx of list, node)
                    <2> after popping out node with min val, push its next node into h
                    
                (2) -:
                    for each iteration, pop out node with min val
            
            2.  dummy:
                (1) add new node: after popping out node in h, add it to end of dummy
        
        Algorithm:
            1.  push all heads into h
            2.  while h:
                (1) pop out node with min val and connect it to dummy
                (2) push its next node into h
        '''
```





### 206. Reverse LinkedList

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        cur = prev.next
        while cur.next is not None:
            temp = cur.next
            cur.next = temp.next
            temp.next = prev.next # notice this is not cur
            prev.next = temp
        return dummy.next
        
        '''
        dummy -> 1 -> 2 - > 3 -> 4
        prev    cur  temp
        
        put temp after prev
        
        dummy -> 2 -> 1 -> 3 -> 4
        prev  (temp) cur
         
        Data Structure:
            1.  prev:
                (1) set: at first, dummy.next = head, prev = dummy
                (2) change next: prev.next = temp
                
            2.  cur
                (1) set: at first, cur = prev.next
                (2) change next: cur.next = temp.next
                
            3.  temp
                (1) set: each iteration, temp = cur.next
                (2) change next: temp.next = prev.next
        
        Algorithm:
            1.  set prev, cur
            2.  while cur.next != None:
                    put temp after prev
                    (Trick: sort the next change by new list, ans is left-bottom
			order:
            temp = cur.next        
			cur.next = 
			temp.next =  
			prev.next =
			
			answer:
			<temp> = cur.next        
			cur.next = [temp.next] 
			[temp.next] = (prev.next)
			(prev.next) = <temp>
        '''
```



#### 92. Reverse LinkedList II

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        for _ in range(left - 1):
            prev = prev.next
            
        cur = prev.next
        for _ in range(right - left):
            temp = cur.next
            cur.next = temp.next
            temp.next = prev.next
            prev.next = temp
        
        return dummy.next
        
        '''
        Explanation:
            1 -> 2 -> 3 -> 4 -> 5
            prev cur  temp
            
            we are going to put temp after pre: 
            (1) 1 -> 3 -> 2 -> 4 -> 5
                prev     cur  temp
                
            (2) 1 -> 4 -> 3 -> 2 -> 5
                prev           cur
        
        Data Structure:
            1.  prev:
                change next: prev.next = temp
            
            2.  cur:
                change next: cur.next = temp.next
                            
            3.  temp:
                (1) set: temp = cur.next
                (2) change next: temp.next = prev.next
        
        Algorithm:
            (1) find pre and cur = prev.next
            (2) iterate (right - left) times:
                    put temp after prev
                
                Trick: sort the next change by new list: temp.next, prev.next, temp
        
        '''
```

#### 25. Reverse Nodes in k-Group

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        cur = prev.next
        while self.ifEnoughNodes(cur, k):
            for _ in range(k - 1):
                temp = cur.next
                cur.next = temp.next
                temp.next = prev.next
                prev.next = temp
            prev = cur
            cur = prev.next
        return dummy.next
    
    def ifEnoughNodes(self, head: Optional[ListNode], k: int):
        while head and k > 0:
            head = head.next
            k -= 1
        if k == 0:
            return True
        else:
            return False
    
    '''
    Explanation:
        Assume k == 2
        dummy -> 1 -> 2 - > 3 -> 4
        prev    cur  temp
        
        After reversing first two nodes, we get:
        
        dummy -> 2 -> 1 -> 3 -> 4
        prev         cur  temp
        
        Then we should reset prev and cur to reverse next pair of nodes
        dummy -> 2 -> 1 -> 3 -> 4
                     prev  cur
    
    Data Structure:
        1.  prev:
                (1) set: 
                    <1> at first, dummy.next = head, prev = dummy
                    <2> after reversing one set of nodes, prev = cur
                (2) change next: prev.next = temp
                
        2.  cur
            (1) set: 
                <1> at first, cur = prev.next
                <2> after reversing one set of nodes, cur = prev.next
            (2) change next: cur.next = temp.next

        3.  temp
            (1) set: each iteration, temp = cur.next
            (2) change next: temp.next = prev.next
    
    Algorithm:
        1.  initialize prev, cur
        2.  while enough nodes for one set to reverse:
                reverse nodes
                reset prev and cur
    '''
```



### 116. Populating Next Right Pointers in Each Node

```python
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        def dfs(node):
            if not node:
                return 
            if node.left:
                node.left.next = node.right
                
            if node.right and node.next:
                node.right.next = node.next.left
            
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        return root
        '''
        Data Structure:
            1.  node: 
                (1) change next of node.left:
                    node.left.next = node.right
                (2) change next of node.right 
                    node.right.next = node.next.left
            
            2.  dfs(node): 
                (1) recursion: after changing next of node.left and node.right, dfs(node.left), dfs(node.right)
        '''
```

#### 117. Populating Next Right Pointers in Each Node II

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':                
        nodeLastRow = root
        while nodeLastRow:
            dummy = Node(0)
            nodeCurRow = dummy
            while nodeLastRow:
                if nodeLastRow.left:
                    nodeCurRow.next = nodeLastRow.left
                    nodeCurRow = nodeCurRow.next
                    
                if nodeLastRow.right:
                    nodeCurRow.next = nodeLastRow.right
                    nodeCurRow = nodeCurRow.next
                nodeLastRow = nodeLastRow.next
            nodeLastRow = dummy.next
        return root
                    
        """
        Data Structure:
            1.  nodeLastRow:
                (1) set:
                    <1> at first row, nodeLastRow = root
                    <2> after iterating all nodes in lastRow, nodeLastRow = dummy.next (firstNode in curRow)
                
            2.  nodeCurRow:
                when iterating nodes in lastRow:
                (1) set node.left to nodeCurRow.next
                (2) set node.right to nodeCurRow.next
        
        Algorithm:
            1.  set root to nodeLastRow
            2.  while nodeLastRow is not None:
                (1) create a dummy node
                    iterating nodes in last row: try to add node.left and node.right to nodeCurRow
                (2) move nodeLastRow to cur Row
        """
```



### 138. Copy List with Random Pointer

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        oldNode2idx = {}
        idx2newNode = {}
        idx = 0
        dummy = Node(0)
        
        oldNode = head
        newNode = dummy
        while oldNode:
            oldNode2idx[oldNode] = idx
            newNode.next = Node(oldNode.val)
            idx2newNode[idx] = newNode.next
            
            oldNode = oldNode.next
            newNode = newNode.next
            idx += 1
        
        oldNode = head
        newNode = dummy.next
        while oldNode:
            if oldNode.random:
                newNode.random = idx2newNode[oldNode2idx[oldNode.random]]
            oldNode = oldNode.next
            newNode = newNode.next
        return dummy.next
        '''
        Data Structure:
            1.  oldNode2idx
                +: during first iteration of old list, oldNode2idx[oldNode] = idx
            
            2.  idx2newNode: 
                +: during first iteration of old list, create a new node out of old node, idx2newNode[idx] = newNode
            
            3.  newNode:
                (1) next: oldNode.next + oldNode2idx + idx2newNode --> newNode.next
                
                (2) random: oldNode.random + oldNode2idx + idx2newNode --> newNode.random
        
        Algorithm:
            1.  copy all nodes with its next pointer:
                (1) update oldNode2idx, idx2newNode
                (2) copy all nodes with its next pointer
                
            2.  copy random pointer:
                (1) update newNode.random
        '''
```



## Union Find



## Binary Search

### 33. Search in Rotated Sorted Array

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        l, r = 0, n - 1
        while l < r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m
            elif nums[l] <= nums[m]:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return l if nums[l] == target else -1
    
    '''
    Explantion:
        4 5 6 1 2 3
        left part: 4 5 6
        right part: 1 2 3
        (1) if mid in on 5, consecutive part [4, 5], discrete part [6, 1, 2, 3]
        (2) if mid is on 2, discrete part [4, 5, 6, 1], consecutive part [2, 3]
         
    Data Structure:
        1.  l:
            abandon nums on the left: 
            (1) if mid is in left part and target is in discrete part
            (2) if mid is in right part and target is in consecutive part
            
        2.  r:
            abandon nums on the right: 
            (1) if mid is in left part and target is in consecutive part
            (2) if mid is in right part and target is in discrete part
            
            
        3.  m:    
            for each iteration: m = l + (r - l) // 2
    
    Algorithm:
        (1) while l < r, do iterations
        (2) for each iteration:
            (1) if mid is on the left:
                <1> if target is in consecutive part
                <2> if target is in discrete part
            (2) if mid is on the right:
                <1> if target is in consecutive part
                <2> if target is in discrete part
    '''
```



## BFS

### 103. Binary Tree Zigzag Level Order Traversal

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque()
        q += [root]
        level = 1
        res = []
        
        while q:
            size = len(q)
            path = []
            for _ in range(size):
                cur = q.popleft()
                if level % 2 == 1:
                    path += [cur.val]
                else:
                    path = [cur.val] + path
                
                if cur.left:
                    q += [cur.left]
                    
                if cur.right:
                    q += [cur.right]
            res += [path]
            level += 1
        
        return res
        '''
        Data Structure:
            1.  q: 
                (1) pop: iterate nodes in one level
                (2) push: adding nodes in next level when popping nodes in one level
            
            2.  level:
                +1: after popping out all nodes in one level
            
            3.  path: +
                (1) if level is odd: path = path + [new node]
                (2) if level is even: path = [new node] + path
            
            4.  res: list of path
                + after iteration of whole level: add path to res
        
        Algorithm:
            1.  initialize q and level
            
            2.  if current level is odd --> add from left to right, 
                otherwise, add from right to left
        '''
```



### 207. Course Schedule

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        n = numCourses
        postCourses = defaultdict(list)
        preCourseNum = defaultdict(int)
        
        for post, pre in prerequisites:
            nextCourses[pre] += [post]
            preCourseNum[post] += 1
        
        q = deque()
        v = set()
        for c in range(n):
            if preCourseNum[c] == 0:
                q += [c]
                v.add(c)
        
        while q:
            size = len(q)
            for _ in range(size):
                c = q.popleft()
                for nc in nextCourses[c]:
                    preCourseNum[nc] -= 1
                    if preCourseNum[nc] == 0 and nc not in v:
                        q += [nc]
                        v.add(nc)
        
        return len(v) == n
                
    '''
    Data Structure:
        1.  nextCourses:
            +: iterate prerequisites [a, b](b -> a): nextCourse[b] += [a]
            -: after reducing preCourseNum, check if its preCourseNum == 0

        2.  preCourseNum:
            +: iterate prerequisites [a, b](b -> a): PreCourseNum[a] += 1
            -: after popping out one course, reduce preCourseNum of all next courses

        3.  q:
            +:  (1) at first, add all courses whose preCourseNum == 0
                (2) after popping out one course and reduce preCourseNum of its next course, check if its preCourseNum == 0
            -:  for each iteration, pop out one course

        4.  v:  
            +:  (1) add course to v when its preCourseNum becomes 0

    Algorithm:
        1.  initialize nextCourses and preCourseNum
        2.  initialize q and v
        3.  for each iteration:
            (1) pop out one course whose preCourseNum == 0
            (2) reduce preCourseNum of its nextCourse
            (3) check if its preCourseNum == 0:
                <1> if so, add it to q, v, delete it from nextCourses
        4.  return true if len(v) == n
        '''
```



#### 210. Course Schedule II

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        n = numCourses
        nextCourses = defaultdict(list)
        preCourseNum = defaultdict(int)
        
        for post, pre in prerequisites:
            nextCourses[pre] += [post]
            preCourseNum[post] += 1
        
        q = deque()
        v = set()
        res = []
        for c in range(n):
            if preCourseNum[c] == 0:
                q += [c]
                v.add(c)
                res += [c]
        
        while q:
            size = len(q)
            for _ in range(size):
                c = q.popleft()
                for nc in nextCourses[c]:
                    preCourseNum[nc] -= 1
                    if preCourseNum[nc] == 0 and nc not in v:
                        q += [nc]
                        v.add(nc)
                        res += [nc]
        
        return res if len(v) == n else []
    '''
    Data Structure:
        1.  nextCourses:
            +: iterate prerequisites [a, b](b -> a): nextCourse[b] += [a]
            -: after reducing preCourseNum, check if its preCourseNum == 0

        2.  preCourseNum:
            +: iterate prerequisites [a, b](b -> a): PreCourseNum[a] += 1
            -: after popping out one course, reduce preCourseNum of all next courses

        3.  q:
            +:  (1) at first, add all courses whose preCourseNum == 0
                (2) after popping out one course and reduce preCourseNum of its next course, check if its preCourseNum == 0
            -:  for each iteration, pop out one course

        4.  v:  
            +:  (1) add course to v when its preCourseNum becomes 0

    Algorithm:
        1.  initialize nextCourses and preCourseNum
        2.  initialize q and v
        3.  for each iteration:
            (1) pop out one course whose preCourseNum == 0
            (2) reduce preCourseNum of its nextCourse
            (3) check if its preCourseNum == 0:
                <1> if so, add it to q, v, delete it from nextCourses
        4.  return true if len(v) == n
    '''
```



## DFS

### 17. Letter Combinations of a Phone Number

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        d = { "2": "abc", "3": "def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
        res = []
        if digits == "":
            return res
        
        def dfs(digits, path, res):
            if len(digits) == 0:
                res += [path]
                return
            digit = digits[0]
            for letter in d[digit]:
                dfs(digits[1:], path + letter, res)
        
        dfs(digits, "", res)
        return res
        '''
        Data Structure:
            1.  dfs(digits, path, res):
            	(1) base case:
            		if len(digits) == 0, add path to res
                (2) recursion:
                    choose 1 letter and add it to path
                    then dfs(digits[1:], new path, res)
            
            2. path:
                +: choose one letter of digit, path += [letter]
            
            3. res
                +: if all digits have been dealt, add path to res 
        '''
```



### 51. N-Queens

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        def isValid(Qs, newQ):
            newQ_idx = len(Qs)
            for Q_idx, Q in enumerate(Qs):
                if newQ == Q or abs(Q_idx - newQ_idx) == abs(Q - newQ):
                    return False
            return True
        
        def dfs(Qs, lines):
            nonlocal res
            if len(Qs) == n:
                res += [lines]
                return 
            for newQ in range(n):
                newLine = ['.']* n
                newLine[newQ] = 'Q'
                newLine = "".join(newLine)
                if isValid(Qs, newQ):
                    dfs(Qs + [newQ], lines + [newLine])
        
        dfs([], [])    
        return res
        '''
        Data Structure:
            1.  dfs(Qs, lines):  
                (1) recursion:
                    choose position for next Q, if it is valid, add it to Qs and lines, then dfs(new Qs, new lines)
                
            2.  Qs:
                (1) invalid newQ 
                    <1> if there exists one Q in Qs that Q == newQ
                    <2> if there exists one Q in Qs that |newQ - Q| == |newQ_idx - Q_idx| 

                
        Algorithm:
            1.  valid(Qs, newQ):
                iterate each Q in Qs
                if newQ is invalid in terms of Q, return false
            
            2.  dfs(Qs, lines):
                if cur_idx == n: 
                    add lines to res and return
                else:
                    iterate pos for newQ from 0 to n - 1:
                        if valid(g + [new pos]):
                            add newQ to Qs, add newLine to lines
        '''
```



### 79. Word Search

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        def dfs(i, j, w):
            if len(w) == 0:
                return True
            
            temp = board[i][j]
            board[i][j] = '#'
            
            res = False
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and board[ni][nj] == w[0]:
                    res |= dfs(ni, nj, w[1:])
            
            board[i][j] = temp
            return res
        
        res = False
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    res |= dfs(i, j, word[1:])
        return res
        '''
        Data Structure:
            1. dfs(i, j, word):
                (1) recursion: iterate neighbor cell(except parent cell) to check if word[1:] match
                (2) return:  if one of recursions can match, return True
                (3) base case: if len(w) == 0, return True
                
            2. board:
                (1) set to # when recursion to avoid dfs back
                (2) set back after recursion
        '''
```

#### 212 Word Search II

```python
from collections import defaultdict
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.word = None
    
    def addword(self, w):
        cur = self
        for c in w:
            cur = cur.children[c] #build and move
        cur.word = w #cur.word = True


class Solution:
    def findWords(self, grid: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for w in words:
            root.addword(w)
        
        res = []
        m, n = len(grid), len(grid[0])
        def dfs(i, j, node):
            nonlocal res
            if node.word:
                res += [node.word]
            
            temp = grid[i][j]
            grid[i][j] = '#'
            
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] in node.children:
                    dfs(ni, nj, node.children[grid[ni][nj]])
            
            grid[i][j] = temp
            
        for i in range(m):
            for j in range(n):
                if grid[i][j] in root.children:
                    dfs(i, j, root.children[grid[i][j]])
            
        return set(res)
        
        """
        Explanation:
            if we search words by word Search I
            TC will be O(m * n * m * n * k)(number of words to search)
        
            we want to solve it within O(m * n * m * n) by Trie
            
        Data Structure:
            1.  Tire:
                (1) children(dict):
                    +: iterate each char in word, each char's childen is next char
                    
                (2) word: 
                    set to True: last char of one word
                
            2.  dfs(i, j, TrieNode of char grid[i][j]):
                (1) recursion: iterate neighbor chars, if char of neighbor cell is in children of cur cell
            
            3.  grid[i][j]
                (1) set to '#': 
                (2) set back:
            
            4.  res:
                +: dfs(i, j, node), node.word == True
        """   
```



### 93. Restore IP Addresses

```python
class Solution:
    def restoreIpAddresses(self, ss: str) -> List[str]:
        res = []
        def isValid(newIP):
            if 0 <= int(newIP) <= 255 and len(str(int(newIP))) == len(newIP):
                return True
            else:
                return False
        
        def dfs(IPs, s):
            nonlocal res
            if len(s) == 0 or len(IPs) == 4:
                if len(s) == 0 and len(IPs) == 4:
                    res += ['.'.join(IPs)]
                else:
                    return
            for i in range(1, min(3 + 1, len(s) + 1)):
                newIP = s[:i]
                if isValid(newIP):
                    dfs(IPs + [newIP], s[i:])
        dfs([], ss)
        return res
        '''
        Data Structure:
            1.  dfs(IPs, s):
            	(1) base case:
                    if len(s) == 0 or len(IPs) == 4:
                    <1> if len(s) == 0 and len(IPs) == 4: add IPs to res
                    <2> else: return
                (2) recursion:
                    get a new ip from s, check it is valid, then dfs(new ips, new s)
                
            2.  newIP:
                isValid: if 0 <= int(newIP) <= 255 and no leading zeros in newIP --> return True
                
        Algorithm:
            1.  isValid(newIP):
            2.  dfs(IPs, s):
                (1) choose one new IP from s
                (2) check if it is valid
                (3) if valid, recursion
        '''
```



### 128. Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        v = set(nums)
        
        @lru_cache(None)
        def dfs(cur_num):
            if cur_num not in v:
                return 0
            
            return 1 + dfs(cur_num - 1)
        
        res = 0
        for num in nums:
            res = max(res, dfs(num))
            
        return res
        '''
        Data Structure:
            1. v: hashset of numbers

            2.  dfs(cur_num): 
            	(1）base case: if cur_num not in v, return 0
                (2) recursion: if cur_num in v, try to find a smaller num
        '''
```



### 200. Number of Islands

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        
        def dfs(i, j):
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == "1":
                    grid[ni][nj] = "0"
                    dfs(ni, nj)
                    
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    grid[i][j] = "0"
                    dfs(i, j)
                    res += 1
                    
        return res
                
        '''
        Data Structure:
            1.  dfs(i, j):
                (1) recursion: 
                    iterate neighbor cells, if land('1'), count as one island(convert to water)
            
            2.  grid[i][j]
                set to "0": before iteration to de-duplicate recursion
            
        Algorithm:
            iteration + dfs
            1.  intialize q with (0, 0) and v
            
            2.  for each cell:  
                    if "1": 
                    (1) set to "0" (2) dfs(i, j)
            
            3.  dfs(i, j):
                for ni, nj:
                    if "1": (1) set to "0" (2) dfs(ni, nj)
                
        '''
```



## Tree



## Sliding Window



## Design

### 146. LRU Cache

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.d = OrderedDict()
        self.k = capacity

    def get(self, key: int) -> int:
        if key in self.d:
            val = self.d[key]
            del self.d[key]
            self.d[key] = val
            return val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            del self.d[key]
            self.d[key] = value
        else:
            if len(self.d) == self.k:
                self.d.popitem(last=False)
            self.d[key] = value

    '''
    Data Structure:
        1.  d: 
            (1) put:
                <1> client.put(k, v): after deleting existing k, put(k, v)
                <2> client.get(k): if k exist in d, delete it and put it back
                
            (2) delete:
                <1> client.put(k, v): if k exists in d, delete it first 
                <2> client.get(k): if k exist in d, delete it first
                <3> client.put(k, v): if putting (k, v) will cause d to exceed capacity, delete item that first comes in
        Algorithm:
        1.  get:
            (1) if key in d: delete the original pair and add the new one to d
            (2) if key not int d: add it to d directy
            
        2.  put:
            (1) if key in d: delete the original pair and add a new one to d
            (2) if key not in d:
                <1> if len(d) == k: pop the first added pair
                <2> if len(d) < k: add directly
    '''
```



## Stack

### 224. Basic Calculator

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        sum_num = num = 0
        sign = 1
        for c in s + '+':
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in ['+', '-']:
                sum_num += sign * num
                sign = 1 if c == '+' else -1
                num = 0
            elif c == '(':
                stack += [sum_num]
                stack += [sign]
                sum_num = 0
                num = 0
                sign = 1
            elif c == ')':
                sum_num += sign * num
                
                prev_sum_sign = stack.pop()
                prev_sum_num = stack.pop()
                sum_num = prev_sum_num + prev_sum_sign * sum_num
                
                sign = 1
                num = 0
                
        return sum_num
        """
        Explanation:
            sum_nums + sign + num
            sum_nums + sign + (...) --> prev_sum_nums + [prev_sign + sum_nums]
            
        Data Structure:
            1.  stack:
                (1) +: after meeting '(', push sum_nums and sign into stack
                
                (2) -: after meeting '(' and updating sum_nums, pop out prev sum_nums and prev_sign
                
                --> stack: [incomplete-sum1, incomp-sum2] sum_num
                incomp-sum1 waits for incomp-sum2 to proceed
                incomp-sum2 waits for sum_num to proceed
                
                
            2.  sum_nums:
                (1) +:
                    <1> meet '+' or '-', update sum_nums with num
                    <2> meet ')', 
                        a. update old sum_nums with num
                        b. after popping out prev_sum_nums and prev_sign
                           update new sum_nums = prev_sum_nums + prev_sign * (old)sum_nums
                (2) reset:
                    <1> meet '(', push sum_nums to stack and reset sum_nums = 0 
                
            3.  sign:
                (1) set to new sign: meet '+' or '-'
                (2) reset to 1('+'): meet '(' or ')'
                
            4.  num:
                (1) +: meet digit, add it to num
                (2) reset to 0: 
                    <1> meet '+' / '-', after adding (old) num to sum
                    <2> meet '(', after updating sum_nums, reset num = 0
                    <3> meet ')', after updating sum_nums, reset num to 0
            
        Algorithm:
            1.  initialize sum_num = 0, sign = 1
            2.  for each char in s + "+":
                (1) if char is a digit:
                    +num
                (2) if char is [+ / -]:
                    +sum_num
                    reset sign, num
                (3) if char is '(':
                    push sum_num, sign
                    reset sum_num, sign, num
                (4) if char is ')':
                    +sum_num with num
                    pop stack --> prev_sum_num, prev_sign
                    +prev_sum_num with sum_num
                    reset sign, num
        """
```

#### 227. Basic Calculator II

```python
class Solution:
    def calculate(self, s: str) -> int:
        if not s:
            return 0
        stack, num, sign = [], 0, '+'
        for c in s + '+':
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in ["+", "-", "*", "/"]:
                if sign == "+":
                    stack += [num]
                elif sign == "-":
                    stack += [-num]
                elif sign == "*":
                    newNum = stack.pop() * num
                    stack += [newNum]
                else:
                    newNum = int(stack.pop() / num)
                    stack += [newNum]
                sign = c
                num = 0
        return sum(stack)

    """
    Explanation:
        1.  difference between int(n / d) and num // d
            int() --> towards zero print(int(-3 / 2)) == -1
            // --> floor print(-3 // 2) == -2
            
        2.  sign + num + (new sign)
    
    Data Structure:
        1.  stack:
            (1) pop: meet new sign, sign == '*' or '/', after popping out prev num, new num = prev num * or / cur num
            (2) push:
                <1> meet new sign, sign = '+' or '-', push +num or -num to stack
                <2> meet new sign, sign = '*' or '/', after calculating new num, push new num to stack
                
        2.  sign:
            set: meet new sign, after popping / pushing, sign = new sign
            
        3.  num:
            (1) +: 
                if meet digit, add it to num
            (2) reset:
                if meet new sign, reset num = 0
    
    Algorithm:  
        1.  for each char:
                (1) if digit:
                    update cur num
                
                (2) if operator or last pos
                    current num is complete
                    <1> if last sign is + or -
                    	push num to stack
                    <2> if last sign is * or /
                       	calculate new Num
                    	push new Num to stack
                    <3> sign = new sign
                    	reset num
        
        2.  res += all of nums in stack
    """
```



#### 772. Basic Calculator III

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        sum_num = num = 0
        s += '+'
        sign = '+'
        i = 0
        while i < len(s):
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == '(':
                num, length = self.calculate(s[i + 1:])
                i += length
            elif c in ['+', '-', ')']:
                if sign == '+':
                    stack += [num]
                elif sign == '-':
                    stack += [-num]
                if c == ')':
                    return sum(stack), i + 1
                num = 0
                sign = c
            i += 1
        return sum(stack)
        """
        Explanation:
            sum_nums + sign + num
            sum_nums + sign + (...) --> prev_sum_nums + [prev_sign + sum_nums]
            
        Data Structure:
            1.  stack:
                (1) +: after meeting '+' / '-'
                
                (2) -: after meeting ')', merge all nums in stack into one num
                
            2.  sign:
            	(1) initialize as '+'
                (1) set to new sign: meet '+' or '-'
                
            3.  num:
                (1) +: meet digit, add it to num
                (2) merge: after meeing '(', reset it to sum of merged nums in '(...)'
                (3) reset to 0 after meeting ')'
            
        Algorithm:
            1.  initialize sum_num = 0, sign = 1
            2.  for each char in s + "+":
                (1) if char is a digit:
                    +num
                (2) if char is [+ / -]:
                    +sum_num
                    reset sign, num
                (3) if char is '(':
                    push sum_num, sign
                    reset sum_num, sign, num
                (4) if char is ')':
                    +sum_num with num
                    pop stack --> prev_sum_num, prev_sign
                    +prev_sum_num with sum_num
                    reset sign, num
        """
```



## Heap



## DP

#### 44. Wildcard Matching

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(1, n + 1):
            if p[j - 1] != '*':
                break
            else:
                dp[0][j] = True
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == p[j - 1] or p[j - 1] == '?':
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
                    
        return dp[m][n]
        '''
        Explanation:
            dp[0][0] = True
            dp[i][j]: s[0] ~ s[i - 1], p[0] ~ p[j - 1]
        
        Data Structure:
            1.  dp[i][j]
                (1) = True:
                    <1> dp[0][0] = True
                    <2> if p[j] == '?', dp[i][j] = dp[i - 1][j - 1] 
                    <3> if p[j] == '*', 
                        a. if delete '*', dp[i][j] = dp[i][j - 1], 
                        b. if match one char with '*', dp[i][j] = dp[i - 1][j]
                        c. if p: * * * * * *, it can match ""
                    <4> if s[i] == p[j], dp[i][j] = dp[i - 1][j - 1] 
                (2) = False
                    otherwise
        
        Algorithm: 
            1.  dp[0][0] = True
            2.  if p is like: * * * * * a a a ...
                match p with "" ==> dp[0][j] = True
            3.  iterate i, j 
                dp[i][j]
        '''
```



## Bit Manipulation



## Greedy



## Sort