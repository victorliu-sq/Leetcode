# TuSimple

## Array/HashTable

### 119. Pascal's Triangle II

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        last = [1]
        
        for i in range(1, rowIndex + 1):
            cur = []
            for x, y in zip([0] + last, last + [0]):
                cur += [x + y]
            last = cur
            
        return last 
        
        """
        左补零，右补零，加一起
        
        Explanantion:
            row1:   [0] r1, r2, ... rn - 1, rn   
            row2:   r1, r2, r3, ...   rn,   [0]
                +
            cur row:    
        
        Data Structure:
            last row
            
        Algorithm:
            for row index in range(1, target + 1):
                
                for x, y in zip([0] + last row, last row + [0])
                    cur += [x + y]
                    
        TC: tow loops --> O(n ^ 2)
        SC: row: last index + 1 --> O(n)
        """
```



### 311. Sparse Matrix Multiplication

```python
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        l1, l2 = [], []
        m, k, n = len(mat1), len(mat1[0]), len(mat2[0])
        res = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(k):
                if mat1[i][j]:
                    l1 += [(i, j, mat1[i][j])]
        
        for i in range(k):
            for j in range(n):
                if mat2[i][j]:
                    l2 += [(i, j, mat2[i][j])]
        
        for r1, c1, v1 in l1:
            for r2, c2, v2 in l2:
                if c1 == r2:
                    res[r1][c2] += v1 * v2
        
        return res
        """
        Data Structure:
            mat1: m, k 
            mat2: k, n
            
            1. res:  m, n
            2.  list of mat1: (i, j, val)
                list of mat2: (i, j, val)
            
        Algotirhm:
            1. Create a new res matrix
            
            2. add non-zero elements in mat1 and mat2 to list
            
            3.  for ele1 in mat1
                    for ele2 in mat2:
                        if col of ele1 == row of ele2:
                            add it to res
        
        SC: n1, n2 --> number of elements in mat1 and mat2
        	mat1: m * k mat2: k * n
        	
        	O(n1 + n2 + m * n)
        	
        TC: O(m * k + k * n + n1 * n2)
        """
```



### 763. Partition Labels

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        leftmost = 0
        rightmost = 0
        res = []
        d = {}
        
        for i, c in enumerate(s):
            d[c] = i
        
        for i, c in enumerate(s):
            rightmost = max(rightmost, d[c])
            if i == rightmost:
                res += [i - leftmost + 1]
                leftmost = i + 1
        
        return res
        
        """
        l1 ~ ln-1: update rightmost
        ln: i == rightmost
        
        Explanation:
            [l1, l2, ... ln]
            l1 ~ ln- 1: cur pos < rightmost pos in set
            ln: cur pos == rightmost pos in set
        
        Data Structure:
            1.	d: {key: letter, val : rightmost position}
            
            2.	leftmost: left most pos in current window
            	rightmost: right most pos in current window
            
        Algorithm:
            1. calculate rightmost pos of each letter
            
            2.for each pos, letter:
            (1) update rightmost pos of current window
            (2) if pos == rightmost:
            	current window is complete
                <1> add length of cur substring to pos
                <2> update leftmost pos
        """
```



### 939. Min Area Rectangle

```python
class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        v = set()
        res = inf
        for x1, y1 in points:
            for x2, y2 in v:
                if (x1, y2) in v and (x2, y1) in v:
                    res = min(res, abs(y1 - y2) * abs(x1 - x2))
            v.add((x1, y1))
        return res if res != inf else 0
        
        """
        Choose top-left, choose bottom right --> fix remaining two points
        
        Explanation:
            given top-left x1, y1 and bottom-right corner x2, y2
            left two coordinates are (x1, y2) (x2, y1)
            
        Data Structure:
            v: hashset of coordinates of previous points
        
        Algorithm:
            for each point:
                (set it as bottom-right corner)
                (1) choose a point from v and set it as top-left corner 
                (2) if left two coordinates of rectangle are in v:
                        update res with current area
                (3) add point to set
                
        SC: n --> total number of points, v --> O(n)
        
        TC: O(n ^ 2)
        """
```



### 953. Verify an Alien Dictionary

```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        d = {}
        for i, c in enumerate(order):
            d[c] = i
        
        def isBigger(w1, w2):
            m ,n = len(w1), len(w2)
            for i in range(min(m, n)):
                c1, c2 = w1[i], w2[i]
                if d[c1] > d[c2]:
                    return True
                elif d[c1] < d[c2]:
                    return False
            return m > n
        
        for i in range(len(words) - 1):
            if isBigger(words[i], words[i + 1]):
                return False
        return True
        
    """
    is Bigger? (1) bigger prefix: yes (2) smaller prefix: no (3) same prefix: check length
    
    Data Structure:
        1. isBigger(w1, w2): return if order of w1 > w2
        
        2. d: {key: letter, val: order}
        
    Algorithm: 
        1.  initialize d
        
        2.  for each wi and wi + 1
                if isBigger(wi, wi + 1)
                    return False
            return True
    
        3.  isBigger(w1, w2)
            (1) for each letter of w1 and w2 in common pos:
                    if c1 > c2: return True
                    elif c1 < c2:  return False
                    else c1 == c2: continue
            (2) if all letters in common pos are equal: 
                    return len(w1) > len(w2)
    
    m, n --> length of words, length of longest word
    SC: d --> O(26)
    TC: two loops: O(m * n)
    """
```

### 1779. Find Nearest Point That Has the Same X or Y Coordinate

```python
class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        index, smallest = -1, math.inf
        for i, (r, c) in enumerate(points):
            dx, dy = x - r, y - c
            if dx * dy == 0 and abs(dx + dy) < smallest:
                smallest = abs(dx + dy)
                index = i
        return index
```



## Union Find

```python
1. parent node: p[node] == node 

2. find parent node:
    find(n):
        if parent[n] == n:
            return n
        else:
            parent[n] = find(parent[n])
       		return parent[n]

3. connect two points:
	p[a] = b
```



### 685. Redundant Connection II

```python

class Solution:
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        k = len(edges)
        p = {}
        for i in range(1, k + 1):
            p[i] = i
            
        def find(n):
            if p[n] == n:
                return n
            p[n] = find(p[n])
            return p[n]
        
        #find 2 edges that indidate 1 child with 2 parents
        d = {}
        cand1, cand2 = None, None
        for p_, c in edges:
            if c in d:
                cand1, cand2 = d[c], [p_, c]
            else:
                d[c] = [p_, c]
        
        if cand1 is None and cand2 is None:
            # there is only one circle, return edge that detects the circle
            for a, b in edges:
                if find(a) == find(b):
                    return [a, b]
                else:
                    p[b] = a
        else:
            # if cand1 can form a circle w/o cand2 --> return cand1
            # else return cand2(cand2 may form a circle w/o cand1)
            for a, b in edges:
                if [a, b] == cand2:
                    continue
                if find(a) == find(b):
                    return cand1
                else:
                    p[b] = a
            return cand2
    """
    Explanation:
        https://leetcode.com/problems/redundant-connection-ii/discuss/108070/Python-O(N)-concise-solution-with-detailed-explanation-passed-updated-testcases
        
	3 resulted situations by reduncdant edge :
    (1) 2 edges that indicate one child with two parents: higher points to lower
    (2) a circle
    (3) a circle whose one edge is (1)
    circle: lower points to higher

    Data Structure:
        1. cand1, cand2 --> 2 edges of the node with 2 parents but one children
        	cand1: first cand, can2: last candidate
        
    Algorithm:
        1. try to find 2 edges that indidate 1 child with 2 parents:
                    
        2.  (1) there is only 1 circle, return edge that detects the circle
           	(2) <1> if cand1 can form a circle w/o cand2 --> return cand1
           	   	<2> return cand2 (if (1) --> any is ok, if (3) must return can2)
         		
         		
    SC: p, cand1, cand2 --> O(n)
    
    TC: (1) iteration of all edges to find cand1 and cand2 --> O(E)
    	(2) iteration of all edges to find circle --> O(E)
    	(3) all find --> O(V)
    	TC: O(v + 2 * E)
    """
```

TC :O(N + N)

SC: O(N)

N: number of vertex and edges

### 684. Redundant Connection

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        p = {}
        k = len(edges)
        for i in range(1, k + 1): 
            p[i] = i # parent of root is root
        
        def find(n):
            if p[n] == n:
                return n
            r = find(p[n])
            p[n] = r
            return r
        
        for n1, n2 in edges:
            r1, r2 = find(n1), find(n2)
            if r1 != r2:
                p[r1] = r2
            else:
                return (n1, n2)               

        """
        Data Strucutre:
            1. parent --> dictionary: {key:node, val: parent}
            
            2. def find(node) --> return root of node
                    if parent[node] == node:
                        return node
                    return dfs(parent[node])
                    
                    --> faster
                    if parent[node] == node:
                        return node
                    root = dfs(parent[node])
                    parent[node] = root
                    
                    return root
            
            3.  def union(node1, node2, parent):
                    (1) find root of node1 and node2
                    (2) if r1 != r2:
                            parent[r1] = r2
                        else:
                            do nothing
        
        Algorithm:
            1. Initialize parent[node] = node --> parent of root is root
            
            2.  for each edge:
                (1) if useful --> the edge can connect two trees --> in def union, r1 != r2:
                        union together
                        
                (2) if useless --> r1 == r2
                        update last useless edge with cur edge
        
        SC: p: O(V)
        TC: find: with compression, sum of all find == O(V)
        	iteration of edges: O(E)
        	--> O(V + E)
        """
```

TC :O(N + N)

SC: O(N)

N: number of vertex and edges

### 1192. Critical Connections in a Network

```python
from collections import defaultdict
class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        g = defaultdict(list)
        depth = [-1] * n
        
        for a, b in connections:
            g[a] += [b]
            g[b] += [a]
        #list of list --> set of tuples
        res = set(map(tuple, map(sorted, connections)))
        
        def dfs(i, d):
            min_nd = inf
            for j in g[i]:
                nd = -1
                if depth[j] == d - 1:
                    continue
                elif depth[j] == -1:
                    depth[j] = d + 1
                    nd = dfs(j, d + 1)
                else:
                    nd = depth[j]
                
                if 0 <= nd <= d:
                    res.remove(tuple(sorted([i, j])))
                
                min_nd = min(min_nd, nd)
            return min_nd
        
        depth[0] = 1
        dfs(0, 1)
        return res
        
        """
        Explanation:
            1.  critical edges + edges in circles = all edges
            
            2.  How to delete all edges in a circle
            
                assume current node is i with depth d and 
                (1) it has a visited neighbor with depth k
                (2) it can connect to (by dfs) a visited node with depth k
                then all edges between node with depth k to node i can be deleted
                
                if j can connect some nodes whose depth <= d, then edge <i, j> is in a circle
                
            
            3.  In terms of 'depth' above,
                it means first depth that we set for a node in the process of recursion
            
            
        Data Structure:
            1.  dfs(cur node, cur depth): 
            	depth of current node has been set(cur depth)
                (1) set depth of all unvisited neighbor nodes
                (2)	discard all neighbor edges if in a circle (and all lower edges in a circle)
                (3) return min depth that current node can connect
            
            2.  depth[i]: -1(not visited) or >= 1 (rank)
                
            3.  graph[i]: list of next node of node i
            
            4.  connections: set of possible res
            
        Algorithm:
            1.  initialize graph
            
            2.  initialize res:
                list of edges --> set of sorted edge tuple
            
            3.  initialize depth: {key: 0, val : 1} else {key: node, val: -1}
            
            4,  dfs(cur node, cur depth):
                    (1) for each neighbor node of cur node:
                    	find smallest depth that neighbor node can reach
                        <1> if it is parent of cur node: depth of next node == cur depth - 1:
                                ignore this next node

                        <2> elif it has been visited: >= 1
                        	j itself is a node whose depth <= cur depth
                                min_depth of j = v[next node]

                        <3> elif it has not been visited: < 0
                        	we check if it can connect with a visited node whose depth <= cur depth
                            --1 set rank of next node = cur depth + 1
                            --2 min_depth of j (along with discarding all lower edges in a circle) = 
                                
                                depth[j] = cur depth += 1
                                dfs(j, cur depth + 1)
                        
                    (2)	<1>	if depth of next node <= cur depth: circle --> discard this edge
                        
                        <2> update min depth that cur node can connect 
                        	with min depth that next node can be or connect
            
                    (3) return min depth of nodes that current node can connect
        
        SC: max depth of recursion tree O(V)
        TC: state of dfs: O(V)
        	sum of iteration of each state: O(E)
        	-->
        	O(V + E)
        """
```



### 721. Accounts Merge

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        p = {}
        def find(n):
            if p[n] == n:
                return n
            p[n] = find(p[n])
            return p[n]
        
        #initialize parents
        for i in range(len(accounts)):
            p[i] = i
        
        #get d1
        d1 = {} #key: email, val: previous root account num
        for i, (_, *emails) in enumerate(accounts):
            for e in emails:
                if e in d1:
                    p1, p2 = find(i), find(d1[e])
                    p[p1] = p2
                d1[e] = i
        
        #get d2
        d2 = collections.defaultdict(list) #key: root account, val: list of emails
        for e, i in d1.items():
            d2[find(i)] += [e] #find(i) and we cannot use p[i]
        
        #get res
        res = []
        for i, emails in d2.items():
            res += [[accounts[i][0]] + sorted(emails)]
        return res
        
        """
        Explanation:
        	res : [account number + list of emails]
        	-->
        	d{key: account number, val: list of emails}
        	-->
        	d{key: email, val: root account number}
        	-->
        	enumerate(emails)
        
        Data Structure:
            1.  union find:
                parent, find, union
                parnet[email] = account num
                
            2.  To connect root accounts with same email together
                d1: {key: email, val: root account(index)}
            
            3.  d2: {key: root account(index), val: list of emails}
            
            4.  res: list of [[name of root account] + [emails of root account]]
        Algorithm:
            1.  initialize parents of each node
            
            2.  for each account
                    for each email (except name -- first element)
                        (1) if email has not showed before
                            d1[email] = cur account
                            
                        (2) else(showed before
                        	we need to connect current root number with previous account number)
                        	
                             <1> connect cur account and account in d
                             <2> d1[email] = root account of cur account
                             
            3.  for each pair of email, account in d:
                    add email to its root account: d2[account]
                    
            4.  for each pair of root account, list of emails:\
                    (1) add name of root account
                    (2) add list of emails
        """
```









## Binary Search

### 4.Median of Two Sorted Arrays

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
            
        n1, n2 = len(nums1), len(nums2)
        
        l, r = 0, n1
        while l <= r:
            x = l + (r - l) //2 
            y = (n1 + n2) // 2 - x
            
            r1 = nums1[x] if x < n1 else inf
            r2 = nums2[y] if y < n2 else inf
            l1 = nums1[x - 1] if x - 1 >= 0 else -inf
            l2 = nums2[y - 1] if y - 1 >= 0 else - inf
            
            if max(l1, l2) <= min(r1, r2): #Notice that there should be equal
                return min(r1, r2) if (n1 + n2) & 1 else (max(l1, l2) + min(r1, r2)) / 2
            elif l1 > r2:
                r = x - 1
            else:
                l = x + 1
        return -1
            
    """
    Binary Search
    Explanation:
    	we need to swap num1 and num2 if len(nums1) > len(nums2)
    	or mid number we choose can be too big
    
    Data Structure:
        1.  left = 0, right = len of nums1
            
            mid, x --> how many nums of half are in nums1
            y --> how many nums of half are in nums2
            half: (l1 + l2 + 1) // 2
    
    Algorithm:
        1,swap num1 and nums2 if len of nums1 > len nums2
        	

        2.left = 0, right = len(nums1)
            x = left + right // 2
            y = (n1 + n2 + 1) // 2 - x

            l1, l2, r1, r2 = nums[...] if not out of boundary else inf / -inf

            if  max(l1, l2) <= min(r1, r2) found
            elif l1 > r2
                right = x - 1

            elif l2 > r3
                left = x + 1
                
    TC:	O(log min(l1, l2))
    SC: O(1)
    """
```





### 1231. Divide Chocolate

```python
class Solution:
    def maximizeSweetness(self, A: List[int], k: int) -> int:
        total, min_s = sum(A), min(A)
        
        def helper(t):
            res = 0
            cur = 0
            for a in A:
                cur += a
                if cur >= t:
                    res += 1
                    cur =0
            return res
        
        l, r = min_s, total
        while l < r:
            m = l + (r - l) // 2 + 1
            if helper(m) > k + 1:
                l = m + 1
            elif helper(m) < k + 1:
                r = m - 1
            else:
                l = m
        return l
        """
        Binary Search
        
        Data Structure:
            1.  left = min chocolate, right = sum of chocolate
                mid = one possible min total sweet
            
            2.  helper(threshold): return max number of pieces of chunks if we divide chocolate greedily into chunks >= threshold
                cur_sum
                cur_piece
            
        Algorithm:
            1.  mid = left + (right - left) // 2 + 1
                
                if helper(mid) == k + 1:
                    try to find a bigger mid
                elif < k + 1:
                    must be smaller
                else:
                    must be bigger --> but can be not that bigger, left = mid
            
            2.  helper(min total sweet):
                for each sweet:
                    (1) cur_sum += sweet
                    (2) if cur_sum > min total sweet:
                            cur_piece += 1
                            cur_sum = 0   
        TC: O(N log sum(A)) N : length of A
        SC: O(1)
        """
```



## DP

### 10.Regular Expression Matching

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        for j in range(2, n + 1, 2):
            if p[j - 1] == '*':
                dp[0][j] = True
            else:
                break
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == p[j - 1] or p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                    
                elif p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]
                    
                    if s[i - 1] == p[j - 2] or p[j - 2] == '.': 
                        dp[i][j] = dp[i][j] or dp[i - 1][j - 2] or dp[i - 1][j]
            
        return dp[m][n]
    
        """
        Data Structure:
            dp[i][j]: Match or not(s[i] can match or not) with i chars in s and j chars in p
            
        Algorithm:
            1.  initialize dp[0][0] = True
            
            2.  initialize dp[0][j] while j % 2 == 0 and p[j - 1] == "*"
            
            3.  Transfer function:
                (1) if s[i - 1] == p[j - 1] or p[j - 1] == '*'
                        dp[i][j] = dp[i - 1][j - 1]
                
                (2) if p[j - 1] == '*':
                        <1>	delete * and previous char
                        	a * --> empty
                        dp[i][j] = dp[i][j - 2] #there will be a previous valid character to match.
                        
                        <2>	try to match
                        if s[i - 1] == p[j - 2] or p[j - 2] == '.':
                          
                         	s: ... a
                         	p: ...a * -->  ... a or ... a * a
                            dp[i][j] = dp[i][j] or dp[i - 1][j - 2] or dp[i - 1][j]
        TC: O(m * n)                    
        SC: O(m * n)                  
        """
```

### 32. Longest Valid Parentheses

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        res = 0
        n = len(s)
        dp = [0] * n
        for i in range(n):
            if s[i] == ')' and i - (dp[i - 1] + 1) >= 0 and s[i - (dp[i - 1] + 1)] == '(':
                dp[i] = dp[i - 1] + 2
                dp[i] += dp[i - dp[i - 1] - 2] if i - dp[i - 1] - 2 >= 0 else 0
            res = max(res, dp[i])
        return res        
        """
        Data Structure:
            dp[i]:  length of longest ()s ending with i
        
            
        Algorithm:
            1.
                i - 1 - dp[i - 1], i - dp[i - 1], i - 1, i
                        (                (    ...     )   )
                
                Notice that there are in total dp[i - 1] ( or ), name points
                
                
                if s[i] == ")" and i - 1 - dp[i - 1] == "(" 
                    dp[i] = 2 + dp[i - 1]
                
                
            2.
                                i - 2 - dp[i - 1]   i - 1 - dp[i - 1]         i
                ()()()...(          )                   (           ...       )       

                if i - dp[i - 1] - 2 >= 0:
                    dp[i] += dp[i - dp[i - 1] -2]
                    
        TC: O(n)
        SC: O(n)
        """
```



### 312. Burst Balloons

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        arr = [1] + nums + [1]
        n = len(arr)
        dp = [[0] * n for _ in range(n)]
        
        for l in range(3, n + 1):
            for i in range(0, n - l + 1):
                j = i + l - 1
                dp[i][j] = max(dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j] for k in range(i + 1, j))
        
        return dp[0][n - 1]
        
        """
        Data Structure:
            arr = [1] + nums + [1]
            dp[i][j]: max score to burst all balloons i + 1 ~ j - 1
        
        Algorithm:
            1. add [1] to start and end of nums
            
            2. Transfer function:
                dp[i][j]
                if k is last balloon to burst --> dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
                
        SC: O(n ^ 2)
        TC: O(n ^ 2)
        ""
```

### 1547. Min Cost to Cut a Stick

```python
class Solution:
    def minCost(self, n: int, A: List[int]) -> int:
        A = sorted(A + [0, n])
        m = len(A)
        
        dp = [[0] * m for _ in range(m)]
        
        for d in range(3, m + 1):
            for i in range(0, m - d + 1):
                j = i + d - 1
                dp[i][j] = min(dp[i][k] + dp[k][j] for k in range(i + 1, j))
                dp[i][j] += A[j] - A[i]
        return dp[0][m - 1]
        
    """
    Data Structure:
        dp[i][j]: merge all sticks from A[i] to A[j]
        
        (1) dp[i][j] = min dp[i][k] + dp[k][j] for k in [i + 1, j - 1]
            Notice that this is different from:
            for k in range():
                dp = min(dp, new dp)
                
        
        (2) merge all stick of dp[i][k] and dp[k][j] --> dp[i][j] += sum [i ~ j] = A[j] - A[i]
        
    Algorithm:
        1.  A = [0] + A + [n]
        
        2.  (1) if d == 2, we have no cost to merge
            (2) for d in range(3, n + 1):
                    for i in range(0, n - d + 1):
                        Transfer function
    
    SC: O(n * n)
    TC: O(n * n)
    """
```



### 361. Bomb Enemy

```python
class Solution:
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        #Left
        self.dp = [[0] * n for _ in range(m)]
        self.res = 0
        self.cur = 0
        
        def helper(x, y):
            c = grid[x][y]
            if c == 'E':
                self.cur += 1
            elif c == 'W':
                self.cur = 0
            else:
                self.dp[x][y] += self.cur
                self.res = max(self.res, self.dp[x][y])
        
        #up
        for j in range(n):
            self.cur = 0
            for i in range(m):
                helper(i, j)
        
        #down
        for j in range(n):
            self.cur = 0
            for i in range(0, m)[::-1]:
                helper(i, j)
        
        #left
        for i in range(m):
            self.cur = 0
            for j in range(0, n):
                helper(i, j)
        
        #right
        for i in range(m):
            self.cur = 0
            for j in range(0, n)[::-1]:
                helper(i, j)
        
        return self.res
        """
        Data Structure:
            preSum[i][j]: number of enemies from left / right / up / down
        
        Algorithm:
            1. iterate each direction
                (1)up
                fix col
                iterate row from up to down
                
                (2)down:
                fix col
                iterate row from down to up
                
                (3)left:
                fix row
                iterate col from left to right
            
                (4)right:
                fix row
                iterate col from right to left
            
            2. update preSum:
                if E:
                    enemy += 1
                elif W:
                    enemy = 0
                else:
                    dp[i][j] = enemy
        
        TC: O(n * n)
        SC: O(n * n)
        """
```



### 877. Stone Game

```python
class Solution:
    def stoneGame(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dp[i][i] = nums[i]
        
        for l in range(2, n + 1):
            for s in range(n - l + 1):
                dp[s][s + l - 1] = max(nums[s] - dp[s + 1][s + l - 1], nums[s + l - 1] - dp[s][s + l - 2])
                
        return dp[0][n - 1] >= 0
        """
        Data Structure:
            dp[i][j] : max (Alice - Bob) in range i ~ j
            
        Algorithm:
            1.  initialize base case with len == 1
                dp[i][i] = nums[i]
            2.  for len in 2 ~ n
                    for i in range(0, n - l + 1):
                        dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
        SC: O(n ^ 2)
        TC: O(n ^ 2)
        """
```

### 1140. Stone Game II

```python
class Solution:
    from functools import lru_cache
    def stoneGameII(self, A: List[int]) -> int:
        n = len(A)
        postSum = [0] * n
        for i in range(0, n)[::-1]:
            postSum[i] = A[i] + (postSum[i + 1] if i < n - 1 else 0)
        
        @lru_cache(None)
        def dfs(s, M):
            if n - 1 - s + 1 <= 2 * M:
                return postSum[s]
            
            res = 0
            for x in range(1, 2 * M + 1):
                temp = postSum[s] - dfs(s + x, max(M, x))
                res = max(temp, res)
            return res
        
        return dfs(0, 1)
        """
        Data Structure:
            1.  dfs(start, M):  return max score of Alice / Bob
                start: index of first pile that we can pick
                M
            
            2.  postSum[i]: sum of num i to num n - 1
            
        Algorithm:
            1.  initialize postSum
                postSum[i] = postSum[i + 1] + nums[i]
                    
            2.  dfs(start, M)
                (1) if num of remaining piles <= M:
                        return sum of piles
                
                (2) choose first x
                    for x in range(1, 2 * m + 1):
                        max Score of Alice  = max(sum of scores -  (max Score of Bob))
                                            = max(postSum[s] - dfs(new s, new M))
                        #score of current person --> s ~ s + x - 1 --> start of next person = s + x
                                            m = max(m, x)
         SC: max depth of recursion tree is O(N)
         
         TC: DFS + DP --> DP is characterized by s and M --> n * n states
         	 we need to iterate N stones for each state --> O(n ^ 3)
         """
```



### 1548. The Most Similar Path in a Graph

```python
class Solution:
    def mostSimilar(self, n: int, roads: List[List[int]], names: List[str], targetPath: List[str]) -> List[int]:
        #initialize g
        g = defaultdict(list)
        
        for u, v in roads:
            g[v] += [u]
            g[u] += [v]
        
        m = len(targetPath)
        dp = [[inf] * n for _ in range(m)]
        prev = [[-1] * n for _ in range(m)]
        
        #initialize dp
        for i in range(n):
            dp[0][i] = names[i] != targetPath[0]
        
        #Transfer Function of dp and set prev
        for d in range(1, m):
            for v in range(n):
                cost = names[v] != targetPath[d]
                for u in g[v]:
                    if dp[d][v] > dp[d - 1][u] + cost:
                        dp[d][v] = dp[d - 1][u] + cost
                        prev[d][v] = u
        
        #find end node with min edit distance
        last, min_cost = -1, inf
        for i in range(n):
            if dp[m - 1][i] < min_cost:
                min_cost = dp[m - 1][i]
                last = i
        
        #re-construct path: m - 1 --> 1
        res = [last]
        for i in range(m - 1, 0, -1):
            last = prev[i][last]
            res += [last]
        return res[::-1]
    """
    Data Structure:
        (1) graph[i]: list of PREVIOUS nodes
        
        (2) dp[cur length][cur end point]: min edit distance with cur length and cur end point
        
            Transfer function:
            dp[k][i] = min (dp[k - 1][j] + name[i] != target[k])
            
        (3) prev[cur lengt][cur end point]: prev point with cur length and cur end point
                
    Algorithm:
        1.  initialize graph
        
        2.  (1) initialize all dp[0][end] = 1 or 0
        
            (2) compute min distance on each pos with each end
                dp[i][v] = min (dp[i - 1][u] for u in graph[v]) + 1 if v != targetPath[i]
                
            (3) update previous end of cur end with min pos
                prev[i][v] = u            
            
        3. re-construct path by prev
            (1) find end with min edit distance
            
            (2) re-construct path from end by prev
                for i in range(0, m)[::-1]
                    path.add(prev[i][end])
                    i -= 1
                    end = prev[i][end]
    """
```



## DFS(Backtracking)

### 22. Generate Parentheses

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        self.res = []
        
        def dfs(left, right, path):
            if left + right == 2 * n:
                self.res += [path]
                return 
            
            if left < n:
                dfs(left + 1, right, path + "(")
            
            if left > right:
                dfs(left, right + 1, path + ")")
        
        dfs(0, 0, '')
        return self.res
        """
        Data Structure:
            dfs(cur left, cur right, cur path): add a left or right to cur path
        
        Algorithm:
            1.  dfs(cur left, cur right, cur path): 
                (1) if left + right == 2 * n:
                        add path to res
                        
                (2) if left < n
                    add left --> dfs(left + 1, right, path + ( )
                
                (3) if left > right:
                        add right --> dfs(left, right + 1, path + ) )
       	
       	SC: depth of recusion Tree O(2 * n) n paris --> n open and n clos
        TC: O(2 ^ (2n))
        """
```

### 39. Combination Sum

```python
class Solution:
    def combinationSum(self, A: List[int], target: int) -> List[List[int]]:
        self.res = []
        n = len(A)
        
        def dfs(s, t, p):
            if t == 0:
                self.res += [p]
                return
            
            for i in range(s, n):
                if A[i] <= t:
                    dfs(i, t - A[i], p + [A[i]])
        
        dfs(0, target, [])
        return self.res
        
        """
        Data Structure:
            dfs(start index, cur target, cur path):
                add num to path
    
        Algorithm:
            1.  if cur target == 0:
                    add path to res
                    return
                    
            2.  for start ~ end of candidates:
                    if cur cand <= target:
                        start = cur index, cur path + [cand]:
                        dfs(new start, new target, new path)
                        
        n: length of arr, t: target
        SC: Assume all nums are 1 --> max depth of recursion tree --> O(t)
        TC: Assume all nums are 1 --> O(1 + n + n ^ 2 / 2 + ... ) --> O(n ^ t)
        
        first level: 1
        second level: n
        third level: n - 1 + n - 2 + ... + 0 --> n ^ 2
        ...
        last level : n ^ d --> n ^ t
        
        """
```

### 128. Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        s = set(nums)
        res = 0
        
        @lru_cache(None)
        def dfs(n):
            if n - 1 in s:
                return 1 + dfs(n - 1)
            else:
                return 1
        
        res = 0
        for n in nums:
            res = max(res, dfs(n))
        return res
        """
        Data Structure:
            dp + backtrack
            d: {num, max len}
            dfs(n): return max length of consecutive seq ending with n
        
        Algorithm:
            1. nums --> hashset
            
            2.  for python, we don't need to create a dictionary and update it manually
                for each num in nums:
                    dfs(num)
                (1) p --> r
                    #if max length of seq ending with num has been put into d
                        #return d[num]
                
                (2) r --> c
                    <1> num - 1 exists in nums
                        res = 1 + dfs(num - 1)
                    <2> if not:
                        res = 1
                    #<3> d[num] = res
        
        SC:
        TC:
        """
```

### 200. Number of Islands

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        def dfs(i, j):
            nonlocal grid
            grid[i][j] = '0'
            
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                    dfs(ni, nj)
        
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    res += 1
                    dfs(i, j)
        
        return res        
        """
        Data Structure:
            dfs(x, y) --> 
            (1) we know current pos is land but not visited yet
            (2) convert current pos from land into water
            (3) try to turn all neighboring lands into water
            
            
        Algorithm:
            1.  for each cell:
                    if cell is land:
                        res += 1
                        dfs(land)
            
            2.  dfs:
                (1) convert current land into water
                	grid[x][y] = 0
                    
                (2) r --> c:
                    for each new pos:
                    	if not out of boundary and land:
                        	dfs(nx, ny)
                        	
        SC: O(m * n) (with help of converting into water)
        TC: O(m * n)
        """
```

### 694. Number of Distinct Islands

```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def dfs(i, j):
            grid[i][j] = 0
            
            res = ""
            for k, (di, dj) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    res += (str(k) + dfs(ni, nj))
            return res + "."
        
        s = set()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    island = dfs(i, j)
                    s.add(island)
        return len(s)
        """
        Data Structure:
            dfs(i, j): 
            (1) current pos is land but not visited yet
            (2) convert current pos from land into water
            (3) try to turn all neighboring lands into water
            
            set: store all distinct islands
            
        Algorithm:
            1.  for each coodinate:
                    if land: 
                        cur land = dfs(x, y)
                        set.add(land)
                        
                return len(set)
                
            2.  dfs(i, j):
                    (1) convert current land into water
                    
                    (2) get string of islands from up, down, left, right
                        4 dfs(i + di, j + dj)
                    (3) islands = 'up' + dfs(up) + 'down' + dfs(down) +    right + ... + cur land(".")
                                    0                   1                      2                   
                        Notice that to make backtracking meaningful, we have to add current land at last
                    (4) return island
        
        SC: O(m * n) (with help of converting into water)
        TC: O(m * n)
        """  
```



### 711. Number of Distinct Islands II

```python
class Solution:
    def numDistinctIslands2(self, grid):
        m, n = len(grid), len(grid[0])
        res = set()
        
        def dfs(i, j, island):
            grid[i][j] = 0
            
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    dfs(ni, nj, island)
            island += [(i, j)]
        
        def normalize(island):
            islands = [[] for _ in range(8)]
            for i, j in island:
                islands[0] += [[i, j]]
                islands[1] += [[-i, j]]
                islands[2] += [[i, -j]]
                islands[3] += [[-i, -j]]
                islands[4] += [[j, i]]
                islands[5] += [[-j, i]]
                islands[6] += [[j, -i]]
                islands[7] += [[-j, -i]]
                
            for island in islands:	#1.we sort each land in island
                island.sort()
                x0, y0 = island[0][0], island[0][1]
                for land in island:
                    land[0] -= x0
                    land[1] -= y0
                    
            islands.sort() #we choose island with min lexicographical sort
            return tuple((r, c) for r, c in islands[0])
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    island = []
                    dfs(i, j, island)
                    nor_island = normalize(island)
                    res.add(nor_island)
        return len(res)
    
    """
    Data Structure:
        1. dfs(x, y): return a list of coordinates of lands in island
        
        2. normalize(list of lands in island): return a normalized list of lands in island
    
    Algorithm:
        1. for each coordinate:
                if land:
                (1) get list of coordinate of lands in island
                (2) normalize that list
                (3) add normalized island to set
            
            return len (set)
            
        2.  dfs:
            (1) add current pos and island and convert land into water
            (2) for each neighbor land:
                    dfs(neightbor)
            
        
        3.  normalize:
            (1) one island(list of coordinates of lands) --> list of eight islands

            (2) for each island:
            	<1> sort lands  by x, y
            	
                <2> each coordinate of land - first coordinate of first land

                <3> sort list of eight island lists

            (4) return first list 
        
        SC: list of islands O(8 * m * n)
        TC: iteration of each island O(m * n) and sort island O(m * n log m * n)
    """
```



### 241. Different Ways to Add Parentheses

```python
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]: 
        @lru_cache(None)
        def dfs(s):
            if s.isdigit():
                return [int(s)]
            
            res = []
            for i, c in enumerate(s):
                if c in ['+', '-', '*']:
                    ls, rs = dfs(s[:i]), dfs(s[i + 1:])
                    res += [eval(str(l) + c + str(r)) for l in ls for r in rs]
            return res
    
        return dfs(expression)
        """
        Data Structure:
            1. dfs(s) : return list of possible results， s: s[0] and s[-1] are digits
            
            2. *d: key = s, val = list of possible results
            
        Algorithm:
            1. p --> r:
                if s is digtis --> return [int(s)]
                
            2. r --> c:
                <1> iterate each operator (index is i)
                        leftList = dfs(s[:i])
                        rightList = dfs(s[i + 1:])
                        
                <2> for left in leftList for right in rightList:
                        add left operator right to res 
                        --> eval(str(left) + c + str(right))
        
        SC: max depth of recursion tree O(len(s))
        TC: state of s --> number of substrings --> 1 + 2 + .. + n --> O(n ^ 2)
        	TC: O(n ^ 3)
        """
```



### 291. Word Pattern II (290. Word Pattern, 890.Find and Replace Pattern)

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        def F(w):
            d = {}
            return [d.setdefault(c, len(d)) for c in w]
        
        return F(pattern) == F(s.split())
        """
        1. Data Structure:
            hashmap: <char, word>
        
        2. Algorithm:
            1. if len of p != len of s:
                    return False
            2. for cur pair <c, w>
                    (1) if c in d and d[c] != w
                            return False
                        if c in d and d[c] == w:
                            continue
                    (2) if c not in d and w in vals:
                            return False
                        if c not in d and w not in val:
                            d[c] = w
                            
        m: number of words, n: max length of word         
        SC: d O(n)
        TC: O(m * n) --> divide into n pieces C_m^n, each slice n to validate --> O(n * C_m^n)
        """
```

### 329. Longest Increasing Path in a Matrix

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        
        @lru_cache(None)
        def dfs(x, y):
            res = 1
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                res = max(res, 1 + dfs(nx, ny) if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] < matrix[x][y] else 0)
            return res
        
        res = 0
        for i in range(m):
            for j in range(n):
                res = max(res, dfs(i, j))
        return res
    
    
    """
    Data Structure:
        dfs(i, j): return length of longest increasing path ending with i, j
    
    Algorithm:
        1. for each i, j:
                dfs(i, j)
        
        2. dfs(cur x, cur y)
                res = max (1 + dfs(cur x + dx, cur y + dy) if nx, ny in boundary and is smaller else 0)
            
            return res
    SC: max depth : O(m * n)
    TC: number of states : m * n
    	TC: O(m * n)
    """
```



### 425. Word Squares (422. Valid Word Square)

```python
class Solution:
    def wordSquares(self, words: List[str]) -> List[List[str]]:
        d = collections.defaultdict(list)
        for w in words:
            for i in range(1, len(w)):
                prefix = w[:i]
                d[prefix] += [w]
        
        self.res = []
        def dfs(squares):
            m, n = len(squares), len(squares[0])
            if m == n:
                self.res += [squares]
                return
               
            prefix = "".join([w[m] for w in squares])
            wl = d[prefix]
            for w in wl:
                dfs(squares + [w])
            
        for w in words:
            dfs([w])
            
        return self.res
        
        
        """
        Explanation:
            How to choose current word by chosen words?
                a   b   [c]   d
                b   c   [d]   e
                [c] [d]  x    x
            for 3rd(2-index) word, the first two char must be --> c d
            --> w0[2] + w1[2]
        
        Data Strucutre:
            1.  d: [key = prefix, val = word]
            
            2.  dfs(squares): squares is a list of chosen words and it is valid square
                (1) add a new word by prefix of current squares
                (2) complete new squares --> dfs(new square)
        
        Algorithm:
            1.  for each word:
                put (prefix, word) to d
                
            2.  for each word:
                    set it as first word in squares
                    dfs(square)
            
            
            3.  dfs(squares):
                    (1) p --> r
                        if enough words in squares:
                            add it to res
                            return
                            
                    (2) r --> c
                        calculate prefix with previously chosen words
                    
                        get corresponding words with prefix
                    
                        dfs(squares + word)
                        
        N: number of words, L: max length of word
        SC: max depth of tree: O(L)
        	size of hashmap: O(L * N) number of prefix of each word : L
        TC: number of squares : N ^ L
        	number of words of each prefix : N
        	O(N * N ^ L)
        """
```



### 489. Robot Room Cleaner

```python
class Solution:
    def cleanRoom(self, robot):
        v = set()
        def dfs(x, y, dx, dy):
            robot.clean()
            for _ in range(4):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in v and robot.move(): #we need to judge visited or not firstly
                    v.add((nx, ny))
                    dfs(nx, ny, dx, dy)
                    robot.turnLeft()
                    robot.turnLeft()
                    robot.move()
                    robot.turnLeft()
                    robot.turnLeft()
                robot.turnLeft()
                dx, dy = -dy, dx
        
        dfs(0, 0, 0, 1)
    """
    Data Structure:
        dfs(x, y, dx, dy):
        clean x, y and neighbor cells and initalize dir as dx, dy
        
        (1) clean x, y 
        (2) move robot forward, (clean all neightbor pos and initalize its dir as before) --> dfs(nx, ny)
        (3) move robot back and initalize its dir as before
        
    Algorithm:
        1. clean x, y

        2. for each dir:
                (1) move forward
                
                (2) for each dir:
                    <1> nx, ny = x + dx, y + dy
                        if nx, ny not in v and robot can move:
                        	clean all neightbor pos and initalize its dir as before
                            dfs(nx, ny, dx, dy)
                    
                    <2> move robot back:
                            turn left, turn left, move
                    
                    <3> reset face of robot
                            turn left, turn left
                
                    <4> robot turn left and dx, dy = -dy, dx
    
    SC: O(m * n) v and max depth of recursion
    TC: O(m * n)
    """
```



### 473. Matchsticks to Square

```python
class Solution:
    def makesquare(self, A: List[int]) -> bool:
        n, total = len(A), sum(A)
        if total % 4 != 0:
            return False
        
        A.sort(reverse = True)
        target = total // 4
        
        @lru_cache(None)
        def dfs(state, s, parts):
            if state == (1 << n) - 1 and parts == 4:
                return True
            if s > target:
                return False
            if s == target:
                return dfs(state, 0, parts + 1)
                
            res = False
            for i in range(n):
                if (1 << i) & state == 0:
                    if s + A[i] > target:
                        break
                        
                    res |= dfs((1 << i) | state, s + A[i], parts)
                    if res:
                        return True
            return res
        
        return dfs(0, 0, 0)
    
    """
    Data Structure:
        1.  dfs(cur state, cur sum, cur parts) : 
                formulate a square
    
    Algorithm:
        1.  if sum of sticks % 4 != 0:
                ignore
        
        2.  p --> r:
            if cur parts == 4:
                return True
            if cur sum > target:
                return False
        
        3.  r --> c:
            if cur sum == target:
                dfs(cur state, cur sum = 0, cur parts += 1)
            
            for each unused stick:
                res |= dfs (update state, cur sum += this stick, cur parts)
        
        
            How to make it faster?
            (1) reverse sticks: from big to small
            (2) if cur + stick[i] > target --> break
            (3) if res: return True
    
    SC:depth of recursion tree + number of states --> O(2 ^ n)
    TC:iteration of each state * number of states --> O(n * 2 ^ n)
    """
            
```

### 698. Partition to K Equal Sum Subsets

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        n, total = len(nums), sum(nums)
        if total % k != 0:
            return False
        
        target = total // k
        
        @lru_cache(None)
        def dfs(cur_s, cur_p, state):
            if cur_p == k and (1 << n) - 1 == state:
                return True
            
            if cur_s > target:
                return False
            
            if cur_s == target:
                return dfs(0, cur_p + 1, state)
            
            res = False
            for i, num in enumerate(nums):
                if not (1 << i) & state:
                    res |= dfs(cur_s + num, cur_p, state | (1 << i))
                    if res:
                        return True
            return res
        
        return dfs(0, 0, 0)
        
        
    """
    Data Structure:
        1.  dfs(cur parts, cur sum, state of nums(bitmask)) 
            bitmask: 100001 --> 0th and 6th nums are used
            
            return True or False
    
    Algorithm:
        1.  p --> r:
            (1) cur parts == k and all nums have been used:  10000000 -1 = 1111111
                    return True
            (2) if cur sum > target:
                    return False
            (3) if cur sum == target;
                    return dfs(cur parts + 1, state --> no change, cur sum = 0)
            
        2.  r --> c:
            iterate each num:
                if current num is not used:
                    add it to current part, set it as used (update state)
                    temp = dfs(new state, new cur sum, cur parts)
                    if temp is True
                        return True
                        
    SC: depth of recursion tree + number of states --> O(2 ^ n)
    TC: O(n * 2 ^ n)
    """
```



### 935. Knight Dialer

```python
class Solution:
    def knightDialer(self, n: int) -> int:
        mod = 10 ** 9 + 7
        
        @lru_cache(None)
        def dfs(i, j, n):
            if n == 0:
                return 1
            temp = 0
            for di, dj in [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]:
                ni, nj = i + di, j + dj
                if 0 <= ni <= 3 and 0 <= nj <= 2 and not (ni == 3 and nj == 0) and not (ni == 3 and nj == 2):
                    temp = (temp + dfs(ni, nj, n - 1)) % mod
            return temp
        
        res = 0
        for i in range(4):
            for j in range(3):
                if not (i == 3 and j == 0) and not (i == 3 and j == 2):
                    res = (res + dfs(i, j, n - 1)) % mod
        return res
        """
        Data Structure:
            dfs(cur x, cur y, cur length): 
            return how many distinct moves we have on current pos with remaining length 
            starting with cur x and cur y
            
        Algorithm:
            1.  for each pos in grid:
                    res += dfs(i, j, n)
            
            2.  dfs(i, j, n):
                (1) if n == 1:
                    return 1
                (2）for di, dj in [8 directions:]
                        ni, nj = i + di, j + dj
                        if valid and not in red cell(* or #):
                            res += dfs(ni, nj, n - 1)
                    return res
                    
        SC : depth of recursion tree + number of states = O(3 * 4 * n)
        TC: O((3 * 4) ^ n)
        """
```







### 996. Number of Squareful Arrays

```python
class Solution:
    def numSquarefulPerms(self, nums: List[int]) -> int:
        d = collections.Counter(nums) 
        n = len(nums)
        def dfs(cur_n, cur_k):
            d[cur_n] -= 1
            res = 0
            if cur_k == 0:
                res = 1
            else:   
                for next_n in d:
                    if d[next_n] >= 1 and int((cur_n + next_n) ** 0.5) ** 2 == cur_n + next_n: 
                        res += dfs(next_n, cur_k - 1)
           	
            d[cur_n] += 1
            return res
        
        res = 0
        for num in d: #we need to iterate each unique num
            res += dfs(num, n - 1)
        return res
        """
        Explanation:
        	same num with different pos are the same --> count freq and we only iterate key num
        	
        Data Structure: 
            1. count: [key: num, val: freq]
            
            2. d: [key, num1, val: num2 that makes num1 + num2 is a prefect square]
            
            3.  dfs(end num, remaining num) and a global variable d: 
            	end num is chosen but not visited(removed from count) yet
            	(1) remove current end num
            	(2) return number of path starting with end num 
        
        Algorithm:
            1.  initialize count
            
            2.  for each key num in count:
                    res += dfs(key num, length = n)
                    
            3.  dfs(end num, remaining num)
            		(1)	count[end num] -= 1
             		
            		(2) <1>	if remaining num == 0:
            				res = 1
            				
            			<2>	else we still need to add other num
            				res = 0
                        	for key num in d[key]:
                        		if d[key] > 0:
                            		res += dfs(key num, remaining num - 1)
                    
                    (3)	count[end num] += 1
        SC: O(N)
        TC: O(N!)
        """
```



### 1278. Palindrome Partitioning III

```python
from functools import lru_cache
class Solution:
    def palindromePartition(self, s: str, k: int) -> int:
        def cost(s):
            i, j = 0, len(s) - 1
            res = 0
            while i < j:
                if s[i] != s[j]:
                    res += 1
                i += 1
                j -= 1
            return res
        n = len(s)
        
        @lru_cache(None)
        def dfs(start, part): #cur_s, cur_part
            if start == n and part == 0:
                return 0
            
            if start == n or part == 0:
                return -1
            
            res = inf
            for i in range(start, n):
                temp = dfs(i + 1, part - 1)
                if temp != -1:
                    res = min(res, temp + cost(s[start : i + 1]))
            return res
    
        return dfs(0, k)
            
            
        """
        Data Structure:
        	1. dfs(start index, remaining parts): return cost to divide s[start ~ end] into remaining parts
        
        	2. cost(string): return cost to convert string into a palindrome
        
        Algorithm:
            1.  cost(string):
                s, e = 0, len - 1
                while s <= e:
                    if s != e: cost += 1
                    s += 1, e -= 1
            
            2.  dfs(start index, left parts):
                    (1) if left parts == 0 and start index == n: return 0
                    
                    (2) if left parts == 0 or start index == n : return -1
                    
                    (3) for end index in range(start index, n - 1):
                            <1> compute cost from start to end      
                            <2> compute next dfs from end + 1
                            <3> if next dfs != -1, update res
        
        SC: max depth of tree + number of states : O(n * k)
        TC: dfs(iteration * number of states == k * N * K) + costs(no duplicate n * n) 
        """
```



## Stack

### 20.Valid Parenthesis

```python
class Solution:
    def isValid(self, s: str) -> bool:
        """
        Data Structure:
            1.  stack: unparented left parenthese
            
            2.  d:{key: right ), val: left (}
            
        Algorithm
            1.  for each cur char:
                (1) if cur is ) and cannot be parented with bottom of stack:  
                        empty or not equal d[c]
                        return False
                    else:
                        pop out stack
                        
                (2) elif cur is (
                        add it to stack
        
            2.  return stack == empty
        """
        d = {"]" : "[", ")" : "(", "}" : "{"}
        stack = []
        for c in s:
            if c in d.values():
                stack += [c]
            else:
                if stack and stack[-1] == d[c]:
                    stack.pop()
                else:
                    return False
        
        return stack == []
   	
    """
    SC: O(n)
    TC: O(n) push in and pop out each elements
    """
```

### 84. Largest Rectangle in Histogram

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        res = 0
        heights = [0] + heights + [0]
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                i2 = stack.pop() 
                i1 = stack[-1]
                i3 = i
                res = max(res, heights[i2] * (i3 - 1 - (i1 + 1) + 1))
            stack += [i]
        return res
    
    
    """
    Explanation:
        1.
        i       j        k
                
        if j is chosen height
        i should be first height smaller than j on the left
        k should be first heigth smaller than j on the right
        
        2.
        stack [h1, h2, ... hn] cur h
        h1 ~ hn all candidates of possible j(first smaller on the left is know but on the right is unknown)
            
        before adding hn to stack, we need to pop out all h if hn < h since h will be first smaller num
        
        in this way, hn-1 < hn and hn-1 will be first smaller num of hn on the left
        
    Data Structure:
    	1.	stack: [h1, h2 ... hn] index of possible middle height
    	
    	2.	to compute h1 --> add 0 before index 0 since there should be a smaller num on the left
    		to compute hn --> add 0 after index n - 1
    
    Algorithm:
        1.  height = [0] + height + [0]
        
        2.  while stack and (stack[-2] < stack[-1]) > cur height
                pop out stack[-1] and update res with it
                
        3.  when we cannot pop anymore (stack[-2] < stack[-1]) < cur
                add current to stack
                
    SC: O(n)
    TC: O(n) push in and pop out each elements            
    """
```

### 85.Maximal Rectangle

```python
class Solution:
    def maximalRectangle(self, A: List[List[str]]) -> int:
        m, n = len(A), len(A[0])
        hs = [0] * n
        res = 0
        for i in range(m):
            for j in range(n):
                if A[i][j] == "1":
                    hs[j] += 1
                else:
                    hs[j] = 0
            res = max(res, self.helper(hs))
        return res
        
    def helper(self, heights):
        stack = []
        res = 0
        heights = [0] + heights + [0]
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                i2 = stack.pop() 
                i1 = stack[-1]
                i3 = i
                res = max(res, heights[i2] * (i3 - 1 - (i1 + 1) + 1))
            stack += [i]
        return res
    
    
        """
        Data Structure:
            1.current histogram
            
            2.stack: potential highest bar

        Algorithm:
            1.  for each row:
                    update histogram
            
            2.  update res with max area in current histogram
            
            3.  compute max area in histogram:
                (1) initialize stack with -1
                    add [0] to end of histogram
                    
                (2) iterate i, num in histogram:
                        <1> while height < histogram[stack[-1]]:
                                curH = bottom of stack
                                curW = bottom of stack + 1 ~ i - 1
                                update res with  cur area
                        <2> add i to stack 
        """
```

### 828. Count Unique Characters of All Substrings of a Given String

```python
class Solution:
    def uniqueLetterString(self, s: str) -> int:        
        #key: letter, val : last of pos
        d = {c : [-1, -1] for c in string.ascii_uppercase}
        res = 0
        for i, c in enumerate(s):
            k, j = d[c]
            res += (i - j) * (j - k)
            d[c] = [j, i]
        
        for c, (k, j) in d.items():
            i = len(s)
            res += (i - j) * (j - k)
        
        return res
        """
        907
        
        Explanation:
            1.
            point1 edge point edge ... edge pointn
            num of points = num of edges + 1
            num of edges = point n - point 1
            
            2.
            A XXX A XXX A
            0     x     n
            
            we want to count how many times can 2nd A show up
                (X..A..X) 
                insert ( into pos between first A and second A
                insert ) into pos between second A and third A
            
            temp = (x - 0) * (n - x)
            
            
        
        Data Structure:
            d[26][2]: 
            d[char][0 / 1] --> last two occurrence index of char
            
        Algorithm:
            1.  initialize last two occurrence of each uppercase letter as [-1, -1]
                -1: make sure any pos can be inserted into
            
            2. iterate each pair of index and char:
                    index of last two occurrence: k, j
                    (1) temp = (i - j) * (j - k)
                        res += temp
                    (2) [k, j] -->[j, i]
            
            3.  iterate each uppercase letter and add count of last letter to res
                    for each letter c:
                        temp = (n - j) * (j - k) (n is length of s)
                        res += temp
        
        OC: histogram + stack: O(n)
        TC: push in and pop out each element in stack: O(m * n)
        """     
```

### 907. Sum of Subarray Minimums

```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        stack = []
        res = 0
        arr = [0] + arr + [0]
        for i, a in enumerate(arr):
            while stack and arr[stack[-1]] > a:
                j = stack.pop()
                k = stack[-1]
                res += (i - j) * (j - k) * arr[j]
            stack += [i]
        return res % (10 ** 9 + 7)
        
        
        """
        828 and histogram
        
        Explanation:
            1.
                        |
            |           |           |
           stack[-2]  stack[-1]     cur
           
           all nums between stack[-2], stack[-1] and stack[-1] and cur > stack[-1]
           calculate how many times can stack[-1] show up in substring:
           insert '(' and ')' : (stack[-1] - stack[-2]) * (cur - stack[-1]) 
           
           2.stack
                stack[c1, c2, ... ,cn-1, cn] c
                c1 ~ cn are candidates of middle num(left smaller num is known but not right smaller)
                
                before adding cn into stack
                cn will pop out all c > cn since cn will be first smaller num on the right
                
                in this way
                cn - 1 < cn and cn-1 will be first num < cn on the left

        
        Data Structure:
           	1. arr: add [0] to start and end of arr
           			to compute first num and last num
            
            2. stack: index of strictly increasing num
        
        Algorithm:
            1.  add [0] to start and end of arr
            
            2.  for each num in new arr:
                    while stack[-1] > cur:
                        (1) update res with stack[-2], stack[-1], cur num
                        (2) pop out bottom of stack
                    (3) add index of cur num to stack
        """
```



### 224. Basic Calculator

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        total, num = 0, 0
        sign = 1
        s = s + '+'
        for i, c in enumerate(s):
            if c.isdigit():
                num = num * 10 + int(c)
                
            elif c in ['+', '-']:
                total += sign * num
                num = 0
                sign = 1 if c == '+' else -1
                
            elif c == '(':
                stack += [total]
                stack += [sign]
                total = 0
                sign = 1
                num = 0
                
            elif c == ')':
                pre_sign = stack.pop()
                pre_total = stack.pop()
                total += sign * num
                total = pre_total + pre_sign * total
                num = 0
                sign = 1
        
        return total
        """
        Explanation:
            sum of ()s in stack + last sign in stack + ( + sum of nums + last sign  current num
            
        Data Structure:
            stack: store sum of previous ()s #用于处理有括号的情况
            sum_nums: store sum of nums
            last sign: before num
            num: sum of current digits
            
        Algorithm:
            1.  initialize sum_num = 0, sign = 1
            2.  for each char in s + "+":
                (1) if char is a digit:
                        add digits to num : num * 10 + digit
                        
                (2) if char is [+ / -]:
                        current num is complete --> add num to sum of nums
                        sign = +1 / -1
                        num = 0
                
                (3) if char is "(":
                        (1) add sum_num to stack
                        (2) add sign to stack
                        sum_num = 0
        """
```



### 227. Basic Calculator II

```python
class Solution:
    def calculate(self, s: str) -> int:
        if not s:
            return 0
        stack, num, sign = [], 0, "+"
        
        s += "+"
        for i in range(len(s)):
            if s[i].isdigit():
                num = num*10+ord(s[i])-ord("0")
            elif s[i] in ["+", "-", "*", "/"]:
                if sign == "-":
                    stack.append(-num)
                elif sign == "+":
                    stack.append(num)
                elif sign == "*":
                    stack.append(stack.pop()*num)
                else:
                    stack.append(int(stack.pop()/num))
                sign = s[i]
                num = 0
        return sum(stack)
    """
    Explanation:
        1.  difference between int(n / d) and num // d
            int() --> towards zero print(int(-3 / 2)) == -1
            // --> floor print(-3 // 2) == -2
    
    Data Structure:
        previous nums        + - / *        123
         stack              (last)sign     num
        
        1.  stack: store previous +num or -num
        2.  sign: operator ahead of num
        3.  num:  current incomplete num
    
    Algorithm:  
        1.  for each char:
                (1) if digit:
                    update cur num
                
                (2) if operator or last pos
                    current num is complete
                    <1> if last sign is + or -
                            stack += [+num / -num]
                    <2> if last sign is * or /
                            stack += [stack[0] * / num]
                    <3> update num and last sign
                        num = 0
                        last sign = cur operator
        
        2.  res += all of nums in stack
    """         
```

### 772. Basic Calculator III

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        num = 0
        s = s + '+'
        sign = '+'
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == '(':
                num, length = self.calculate(s[i + 1:]) #(1) num = res (2) i + 1
                i += length
            else:
                if sign == '+':
                    stack += [+num]
                elif sign == '-':
                    stack += [-num]
                elif sign == '*':
                    stack += [stack.pop() * num]
                elif sign == '/':
                    stack += [int(stack.pop() / num)]
                if c == ')':
                    return sum(stack), i + 1
                else:
                    num = 0
                    sign = c
            i += 1
        return sum(stack)
        """
        Explanation:
        	+num1, -num2, +num3 ... + num n, sign + cur num
            stack  stack   stack
                    
            By calculate res of string in () recursively
            Convert this problem into Calculator with only +-*/
            
        Data Structure:
            1.  stack:
                all previous complete +num / -num in a single()
                
            2.  last sign
            
            3.  num: current incomplete num
            
            4.  calculate(   " ... )"    ): return (1) res of ... (2) length of ...)
                "...)" comes from ("(>>>)")
        
        Algorithm:
            while i < n:
            1.  if char is digit:
                update num
                
            2.  elif char is '(':
                    res, length = dfs(s[i + 1:])
                    (1)num = res
                    (2)i += length (stand on ')')
                    
            3.  elif char is an operator / ')':
                    (1) calculate current num with last sign and add it to stack
                    #############################################################
                    (2) if char is ')':
                            return sum of stack, length = i + 1
                    #############################################################
                    (3) num = 0, last sign = cur operator
                    
            i += 1
        """
```



### 853. Car Fleet

```python
class Solution:
    def carFleet(self, target, pos, speed):
        time = [(target - p) / s for p, s in sorted(zip(pos, speed))]
        later = 0
        res = 0
        for t in time[::-1]:
            if t > later:
                res += 1
                later = t
            
        return res
    
    """ 
    Data Structure:
        1. [index, (pos, speed)], sorted by pos
        
        2. [index, time] same sort as above
        
        3.  time of latter car fleet
    
    Algorithm:
        1.  calculate[index, time]
        
        2.  iterate each time from back to forth:
                if time > time of right car fleet:
                    res += 1
                    update time of right car fleet = cur time
    """
```



### 1776. Car Fleet II

```python
class Solution:
    def getCollisionTimes(self, cars: List[List[int]]) -> List[float]:
        stack = []
        n = len(cars)
        time = [-1] * n
        for i in range(n)[::-1]:
            p, s = cars[i]
            while stack and (s <= cars[stack[-1]][1] or (time[stack[-1]] > 0 and (cars[stack[-1]][0] - p) / (s - cars[stack[-1]][1]) > time[stack[-1]])): 
                #notice that if prev speed >= cur speed, we cannot catch up
                #if prev time < cur time, we cannot catch up
                stack.pop()
            
            if stack:
                time[i] = (cars[stack[-1]][0] - cars[i][0]) / (cars[i][1] - cars[stack[-1]][1])
            else:
                time[i] = -1
                
            stack += [i]
        return time
        """
        Explanation:
            1.  
            stack: [car1, car2, car3, car4] cur car
            car1 ~ car4 are candidates for cur car to catch up
            before we add car4, we pop out all cars that will not collide with car4
            that means car4 must collide with car3 or car4 is a head of car fleet
            
            In this way, 
            if speed of cur car > car4 or time to collide with 4 > collision time of car4:
            	pop out car4
                
        Data Structure:
            1.  stack:  index of car1, car2, .. car4
            
            2.  time: time[i] is time to collide with next car for car i
            
        Algorithm:
            iterate each car from back for front:
                (1) delete all cars not to collide with cur car in stack:
                    if speed or time of collision not satisfied:
                        pop
                        
                (2) if stack is empty:
                		current car will be head of car fleet
                        time[i] = -1
                    else:
                    	current car will collide with stack[-1] car
                        time[i] = time collision with stack[-1]
                
                (3) add index of cur car to stack
        """
```



### 856. Score of Parentheses

```python
class Solution:
    def scoreOfParentheses(self, s):
        cur = 0
        stack = []
        for c in s:
            if c == '(':
                stack += [cur]
                cur = 0
            else:
                cur = max(2 * cur, 1)
                cur = stack.pop() + cur
        return cur
        """
        Explanation:
           ( ()()	( 	()()    (   ()()
            stack		stack       cur
        
        Data Structure:
            1.  stack: stores sum of previous ()s 
            
            2.  cur: store sum of cur ()s in a incomplete ()
            
        Algorithm:
            iterate each char:
            (1) if (
                store cur in stack
                cur = 0
            
            (2) if )
                (1) cur * 2 or 1
                (2) add cur to stack.pop
        """
```



### 1081.Smallest Subsequence of Distinct Characters

```python
class Solution:
    def smallestSubsequence(self, s: str) -> str:
        stack = []
        v = set()
        d = {}
        for i, c in enumerate(s):
            d[c] = i
            
        for i, c in enumerate(s):
            if c in v:
                continue
            while stack and d[stack[-1]] > i and c < stack[-1]:
                v.remove(stack.pop())
            stack += [c]
            v.add(c)
        
        return "".join(stack)
        """
        Data Structure:
            1.  stack: [c1, c2, ... cn] c1 ~ cn are unique chars
            
            before pushing cn into stack
            we will pop out all stack[-1] which is not unique and > cn
            
            In this way
            cn - 1 is either smaller than cn or cn - 1 is last letter so we cannot pop
            
            2.  v: hashset storing chars in stack
            3.  d: {key: char, val: last pos}
            
        Algorithm:
            for each char:
            1.  if c in stack: prev pos is better than current pos --> continue
            
            2.  while cn is not unique and bigger than current char:
                    remove it from hashset and stack
                    
            3.  add current char to stack
            
        TC:for each char, push in and pop out --> O(2n)
        SC:d: n, v: n, stack: n --> O(n)
        """
```





### 1190. Reverse Substrings Between Each Pair of Parentheses

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        cur = ''
        stack = []
        for c in s:
            if c == '(':
                stack += [cur]
                cur = ''
            elif c == ')':
                cur = stack.pop() + cur[::-1]
            else:
                cur += c
        return cur
        
        """
        stack
        Explanation:
        	( s1 ( s2 ... sn ( cur
        	s1 ~ sn are strings in stack
        	cur is current string
        	
        Data Structure:
        1.  stack: string behind '('
        
        2.  cur: string of current incomplete ()
        
        
        Algorithm:
        for each char:
            if letter:
                add it to cur
            elif (:
                add cur to stack
                cur = ''
            elif ):
                reverse cur
                add cur to stack.pop
        
        TC： O(n ^ 2)
        """
```

### 1249. Min Removes to make valid Parentheses

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        cur = ''
        for c in s:
            if c == '(':
                stack += [cur]
                cur = ''
            elif c == ')':
                if stack:
                    cur = '(' + cur + ')'
                    cur = stack.pop() + cur
            else:
                cur += c
        
        while stack:
            cur = stack.pop() + cur
        return cur
        """
        Data Structure:
            '(' + prev unpaired string + '(' + cur unpaired string 
             2	       stack               1			cur
            	       
           	res:
            
            1. stack: 	store all previous unpaired stings
            			Notice that we meet '(' 1 and push prev unpaired string into stack       
            			
            2. cur: current incomplete string
            
        Algorithm:
            1.  for each char:
                (1)if not ():
                    add it to cur

                (2)elif (:
                    store cur in complete (... in stack
                    cur = ''

                (3)else ):
                    if stack is empty: 
                    	current string is not a unpaired string(no '('), so we just delete cur ')'
                    else:
                        cur = ( + stack.pop + )
                        cur = stack.pop + cur
            
            2.  while stack is not empty:
            		there are many unpaired strings ( ‘(' + s1 + '(' + s2 + ...)
            		
                    corresponding ( is invalid and we just delete all of them
                    cur = stack.pop + cur
        """
```



## BFS

### 127. Word Ladder

```python
class Solution:
    from collections import deque
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        q = deque()
        wordList = set(wordList)
        
        q += [beginWord]
        level = 1
        
        while q:
            size = len(q)
            for _ in range(size):
                w = q.popleft()
                if w == endWord:
                    return level
                for i in range(len(w)):
                    for j in range(26):
                        c = w[i]
                        nc = chr(ord('a') + j)
                        nw = w[:i] + nc + w[i + 1:]
                        if nw in wordList:
                            q += [nw]
                            wordList.remove(nw)
            level += 1
        return 0
        """
        Data Structure:
            q, queue: all the words in the same level
            v, visited: --> delete from wordList
            
        Algorithnm:
            1. add beginWord to q:
            2. while queue is not empty:
                    check each word in the same level / q
                    
                    (1) word is endWord:
                            return level
                            
                    (2) not:
                        for each char in word:
                            char --> 'a' ~ 'z'
                            if new word in wordList
                                add newWord to queue
                                delete newWord from wordList
                    (3) after checking all words:
                            level += 1
        
        N: number of words in wordList
        L: length of word

        TC: O(N * L * L * 26)

        OC: size of v and size of q: O(N * L) # space complexity of a string is O(N)
        	Notice that Space complexity of a string is its length O(L) rather than O(1) like bitmask
        """        
```



### 126. Word Ladder II

```python
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        d = {beginWord: [[beginWord]]}
        wordList = set(wordList)
        res = []
        
        while d:
            nd = collections.defaultdict(list)
            for w in d:
                if w == endWord:
                    res = d[w]
                    return res
                for i in range(len(w)):
                    for j in range(26):
                        c = w[i]
                        nc = chr(ord('a') + j)
                        nw = w[:i] + nc + w[i + 1:]
                        if nw in wordList:
                            np = [p + [nw] for p in d[w]] #Notice that we may meet nw before
                            nd[nw] += np
            wordList -= set(nd.keys())
            d = nd
        return res     
    """
    Data Structure:
        d : [w, list of path(endWord is w)]
        wordList(visited)
        nd: new dictionary to store new word and new path
        
    Algorithm:
        1. add begin word to d
            path : [begin]
            list of path: [[begin]]
            
            {beginWord, [[begin]]}
            
        2.  while d is not empty:
                iterate all words of the same level --> d.keys()
                (1) if w is equal to endWord:
                        add all path to res
                    
                (2) not:
                    <1> find its next word / new word
                    <2> path --> path + new word
                    <3> add [new path list] to nd[new word]
                
                (3) after iteration of all words in current level
                    <1> delete all of new words from wordList
                    <2> update d = nd
    
    N: number of words in wordList
    L: length of word
    SC: size of hashmap + size of set (N * L)
    TC: O(N * L * L * 26)
    """
```



### 207. Course Schedule

```python
class Solution:
    from collections import deque, defaultdict
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        q = deque()
        g = defaultdict(list)
        v = set()
        d = defaultdict(int)
        
        for (a, b) in prerequisites:
            d[a] += 1
            g[b] += [a]
        
        for i in range(numCourses):
            if d[i] == 0:
                q += [i]
                v.add(i)
                
        while q:
            size = len(q)
            for _ in range(size):
                cur = q.popleft()
                for next_c in g[cur]:
                    if d[next_c] >= 1:
                        d[next_c] -= 1
                        if d[next_c] == 0 and next_c not in v:
                            q += [next_c]
                            v.add(next_c)
        
        return len(v) == numCourses
            
        
        """
        Data Structure:
            1. queue --> q
            
            2. graph / next nodes dictionary --> g {key = course, val = list of next courses}
            
            3. degree of nodes --> d
            
            4. visited --> v
        
        Algorithm:
            1. initialize graph and degree:
                    for each pair in prerequisites:[a, b] b --> a
                        degree[a] += 1
                        graph[b] += [a]
                        
            2.  add all nodes with degree == 0 to q, v
            
            3.  while q:
                    pop out all nodes of the same level
                    (1) for each next node of cur node:
                            degree[next] -= 1
                    (2)     if degree[next] == 0 and next not in visited:
                                add next to queue
                                add next to visited
                            
            4.  return len(visited) == numCourses
            
        TC: O(V + E)

		SC: O(V + E)
        """ 
```

### 210. Course Schedule II

```python
class Solution:
    from collections import deque, defaultdict
    def findOrder(self, n: int, relations: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        d = [0] * n
        q = deque()
        v = set()
        
        res = []
        
        #b -> a
        for a, b in relations:
            g[b] += [a]
            d[a] += 1
        
        for i in range(n):
            if d[i] == 0:
                q += [i]
                v.add(i)
                res += [i]
                
        while q:
            size = len(q)
            for _ in range(size):
                i = q.popleft()
                for j in g[i]:
                    if d[j] >= 1:
                        d[j] -= 1
                        if d[j] == 0 and j not in v:
                            q += [j]
                            v.add(j)
                            res += [j]
        
        return res if len(v) == n else []
```

TC : O(V + E)

SC: O(V + E)



### 269. Alien Dictionary

```python
from collections import defaultdict
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        g = defaultdict(list)
        d = defaultdict(int)
        for w in words:
            for c in w:
                g[c] = []
                d[c] = 0
        
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            for j in range(min(len(w1), len(w2))):
                c1, c2 = w1[j], w2[j]
                if c1 != c2:
                    g[c1] += [c2]
                    d[c2] += 1
                    break
                
                if j == min(len(w1), len(w2)) - 1 and c1 == c2 and len(w1) > len(w2):
                    return ""
        
        q = collections.deque()
        v = set()
        res = ""
        
        for c in d:
            if d[c] == 0:
                q += [c]
                v.add(c)
                res += c
                
        while q:
            size = len(q)
            for _ in range(size):
                i = q.popleft()
                for j in g[i]:
                    if d[j] >= 1:
                        d[j] -= 1
                        if d[j] == 0 and j not in v:
                            q += [j]
                            v.add(j)
                            res += j
        
        return res if len(v) == len(d) else ""
    
    """
    Explanation:
            1.  Try to convert words into a graph with g, d
                what information can w1 < w2 give us?
                (1) if w1[j] != w2[j] for the first time
                        c1 < c2 --> c1 ---> c2 
                        
                (2) if all characters in one word are the same
                        abc
                        abcd
                        
                        abc are equal
                        
                    Given Order of alien dictionary:
                        w1 < w2
                    but if len(w1) > len(w2) --> alien dictionary is invalid
                
            2.  validate a course schedule
            
    Data Structure:
        1. g[i]: list of next nodes of node i, key : only letters in alien words
        
        2.  d[i]: num of previous nodes of i, key: only letters in alien words
        
        3.  q: just added nodes whose degree == 0
        
        4.  v: set of added nodes
    
    Algorithm:
        1.  initialize g and d:
            (1) for each char in g:
                    g[char] = []
                    d[char] = 0 
            
            (2) for each pair of w[i] and w[i + 1]:
                <1> for each pair of c1 and c2:
                        if c1 != c2:
                            g[c1] += [c2], d[c2] += 1
                            break
                <2> if all pair of c1 and c2 are equal and len of w[i + 1] > w[i]:
                        g and d cannot be initialized: return empty string
        
        2.  (1) add all nodes whose degree == 0 to queue
            (2) while q:
                <1> pop out all nodes of same level
                <2> for each neighbor node j:
                        d[j] -= 1
                        if d[j] == 0 and not in v:
                            add j to queue, v
        
    """
```

TC : O(V + E) == O(N) number of words

SC: O(V + E) == O(N)



### 261. Graph Valid Tree (Validate Undirected Tree)

```python
class Solution:
    from collections import defaultdict, deque
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False
        
        if len(edges) == 0:
            return True
        
        g = defaultdict(list)
        d = defaultdict(int)
        q = deque()
        v = set()
        
        for a, b in edges:
            g[a] += [b]
            g[b] += [a]
            d[a] += 1
            d[b] += 1
        
        for i in range(n):
            if d[i] == 1:
                q += [i]
                v.add(i)
                
        while q:
            size = len(q)
            for _ in range(size):
                i = q.popleft()
                for j in g[i]:
                    if d[j] >= 1 and d[i] >= 1:
                        d[j] -= 1; d[i] -= 1
                        if d[j] == 1 and j not in v:
                            q += [j]
                            v.add(j)
                        
        return len(v) == n
                
            
        """
        Explanation:
        	start from leaves
        	cut off all leaves and add new leaves to queue
        	
        	check if all nodes can be iterated
        
        Data Structure:
            graph: neighbor nodes
            degree: number of neighbor nodes
            queue
            visited
            
        Algorithm:
            1. if len of edges != len of nodes - 1
                    False
                    
                if len of edges == 0 (len of nodes == 1)
                    return True
            
            2. initialize graph and degree(undirected graph)
            
            3. add all leaves (nodes whose degree == 1) to q and v
            
            4. while q
                (1) pop out all nodes of the same level
                
                (2) degree of neighbor nodes -= 1
                
                (3) if degree == 1 and not visited
                        add neightbor nodes to q and v
        """         
```

TC : O(V + E)

SC: O(V + E)

### 1361. Validate Binary Tree Nodes (Valid Directed Tree)

```python
class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        r = 0
        children = set(leftChild + rightChild)
        for i in range(n):
            if i not in children:
                r = i
        
        v = set()
        q = collections.deque()
        q += [r]
        v.add(r)
        
        while q:
            p = q.popleft()
            if leftChild[p] != -1:
                lc = leftChild[p]
                if lc in v:
                    return False
                else:
                    q += [lc]
                    v.add(lc)
                
            if rightChild[p] != -1:
                rc = rightChild[p]
                if rc in v:
                    return False
                else:
                    q += [rc]
                    v.add(rc)
        return len(v) == n
        """
        Explanation:
            v: store visited nodes
            if children of a node has been put into v, same node has diff parents --> False

        Data Structure:
            1.  set of children --> find root

            2.  queue: visited parents
                v: set of visited children

        Algorithm:
            1.  initialize set of children, find root, and add root to queue and v
				#Notice that we only add one root to queue
            2.  while q:
                    (1) pop out a parent
                    (2) for each children:      
                        <1> if in v: return False
                        <2> put it in q and v
        """
```

TC : O(V + E)

SC: O(V + E)



### 773. Sliding Puzzle

```python
class Solution:
    def slidingPuzzle(self, grid: List[List[int]]) -> int:
        target = "123450"
        start = ""
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                c = grid[i][j]
                start += str(c)
        
        level = 0
        q = deque()
        v = set()
        
        q += [start]
        v.add(start)
        level = 0
        
        while q:
            size = len(q)
            for _ in range(size):
                s = q.popleft()
                if s == target:
                    return level
                idx = s.index('0') #time complexity is m * n
                i, j = idx // n, idx % n
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        ns = [c for c in s]
                        nidx = ni * n + nj
                        ns[idx], ns[nidx] = ns[nidx], ns[idx]
                        ns = "".join(ns)
                        if ns not in v:
                            q += [ns]
                            v.add(ns)
            level += 1
        return -1
        
        """
        Explanation:
            q += [string] m * n
            
            find idx of '0' --> i : idx // n, j: idx % n
            
            i, j --> ni, nj
            
            new_idx = ni * n + nj
            
        Data Structure:
            1.  q : store all puzzle strings of the same level
            
            2.  target: "123450"
            
            3.  level : current level
            
        Algorithm:  
            1.  initialize q: add board
                level = 0
                v.add(board)
                
            2.  while q:
                (1) pop out all strings of the same level
                (2) if string == target, return level
                (3) find pos of '0' in string and convert it into i, j
                    for ni, nj: #npos = ni * n + nj
                        <1> convert string into list
                        <2> swap chars in pos and new pos of list
                        <3> convert new list to new string
                        <4> if new string not in v:
                            add it to queue
                            add it to v
                (4) after iteration of all strings of same level, level += 1
        TC: O(MN * (MN)!)
        SC: O(MN * (MN)!)
        """
```



TC:((M * N )!)

SC:((M * N )!)



### 847. Shortest Path Visiting All Nodes

```python
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        q = deque()
        v = set()
        
        for i in range(n):
            q += [(i, 1 << i)]
            v.add((i, 1 << i))
            
        level = 0
        while q:
            size = len(q)
            for _ in range(size):
                i, s = q.popleft()
                if s == (1 << n) - 1:
                    return level
                for j in graph[i]:
                    ns = s | (1 << j)
                    if (j, ns) not in v:
                        q += [(j, ns)]
                        v.add((j, ns))
            level += 1
        return -1
        """
        Data Structure:
            q: (end node, state of nodes)
            v: {key: node, val: hashset of states}
        
        Algorithm:
            1.  initialize q and v:
                q: (i, 1 << i)
                v: (i, set(1 << i))
                
            2.  while q:
                (1) pop out all ndoes of the same level
                (2) if the state shows that all nodes have been visited:
                        return level
                (3) for next node in graph[ndoe]:
                        if next node not in v:
                            add (next node, new state) to q
                            add new state = v[next node]
                (4) level += 1
        
        TC:O(n * n * 2 ^ n)
        
		SC:O(n * 2 ^ n)
        """
```



### 864. Shortest Path to Get All Keys

```python
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        keys = 0
        m, n = len(grid), len(grid[0])
        t = 0 # t means total number of keys
        
        q = deque()
        v = set()
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '@':
                    q += [(i, j, 0)]
                    v.add((i, j, 0))
                elif grid[i][j] in string.ascii_lowercase:
                    t += 1
        
        level = 0
        while q:
            size = len(q)
            for _ in range(size):
                i, j, k = q.popleft()
                if k == (1 << t) - 1:
                    return level
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] != '#':
                        nk = k
                        if grid[ni][nj] in string.ascii_lowercase:
                            nk = nk | (1 << (ord(grid[ni][nj]) - ord('a')))
                        elif grid[ni][nj] in string.ascii_uppercase:
                            if (1 << (ord(grid[ni][nj]) - ord('A'))) & nk == 0:
                                continue
                        
                        if (ni, nj, nk) not in v:
                            q += [(ni, nj, nk)]
                            v.add((ni, nj, nk))
            level += 1
        return -1
        """
        Explanation:
        	node: <pos, state of keys
        
            for next pos:
            	if key --> get it and empty cell
            	if block --> if key, break it and empty cell else continue
            	
        Data Structure:
            1.  q: (state of key, x, y)
            2.  v: visited(state of key, x, y)
        
        Algorithm:
            1.  iterate each pos:
                    calculate number of key and add start pos and state of key to q
            
            2.  while q
                (1).  pop out all (key, x, y) of the same level
                (2).  if key == 1 << n - 1, return level
                (3).  for each neighbor position:
                        <1> if invalid or wall: continue
                        <2> if key: update key
                            elif lock:
                                if key is valid: move
                                else: continue
                        <3> if (new key, nx, ny) not in v:
                                add it to q
                                add it to v
                (4).  level += 1
        
        TC : O(m * n * 2 ^ k)

		SC: O(m * n * 2 ^ k)
        """
```



### 886. Possible Bipartition

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = collections.defaultdict(list)
        v = [0 for _ in range(n)]
        color = 1
        for a, b in dislikes:
            g[a - 1] += [b - 1]
            g[b - 1] += [a - 1]
        
        for i in range(n):
            if v[i]:
                continue
            q = collections.deque()
            q += [i]
            v[i] = 1
            color = 2
            while q:
                size = len(q)
                for _ in range(size):
                    i = q.popleft()
                    for j in g[i]:
                        if v[j] and v[j] != color:
                            return False
                        if v[j] == 0:	#Notice only uncolored nodes should be in queue
                            v[j] = color
                            q += [j]
                color = 2 if color == 1 else 1
        return True
                    
        """
        Explanation:
        	when we start to color from a node, all nodes in this tree will be iterated
        	
        	so we can start with any color in a new tree and there will be no influence at all
        
        Data Structure:
            1. q: just added nodes of same color
            2. v: v[i], color of node i
            3. color: current color
            4. graph[i] : list of neighbor nodes
        
        Algorithm:
            1.  iterate each node(including all nodes in its tree):
                    if painted before: ignore
                    else:
                        2.  (1) initialize queue with current node
                                color = 1
                                v[cur node] = color
                            (2) while q:
                                    <1> pop out all nodes of same color
                                    <2> for each neightbor node:
                                            <2.1>   if parinted before and different color from cur color
                                                        return False
                                            <2.2>   if not visited:
                                                        add it to queue
                                                        add it to visited, paint it with color
        TC: O(V + E)

		SC: O(V + E)
        """
```



### 934. Shortest Bridge

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        q = collections.deque()
        v = set()
        m, n = len(grid), len(grid[0])
        
        def dfs(i, j): #Notice that we only know i, j is land it has not been put into q and v yet
            nonlocal q, v
            q += [(i, j)]
            v.add((i, j))
            
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1 and (ni, nj) not in v:
                    dfs(ni, nj)
        
        #initialize q and v
        found = False
        for i in range(m):
            if found:
                break
            for j in range(n):
                if grid[i][j] == 1:
                    dfs(i, j)
                    found = True
                    break
        
        #bfs
        print(v)
        level = 0
        while q:
            size = len(q)
            for _ in range(size):
                i, j = q.popleft()
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1 and (ni, nj) not in v:
                        return level
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 0 and (ni, nj) not in v:
                        q += [(ni, nj)]
                        v.add((ni, nj))
            level += 1
        """
        Data Structure:
            q: all lands just added of the same level
            v: visited nodes
            dfs(x, y): push all 1 neighboring x, y into q and v
        
        Algorithm:
            1.  initialize q and v with dfs
                iterate each pos
                    if it is land: dfs(current position)
                    break from entire loop
            
            2.  while q:
                (1) pop all nodes of same level
                (2) for each neighbor node:
                        if unvisited land:
                            return level
                        if unvisited water:
                            add it to q
                            add it to v
            
            3.  dfs(i, j)
                (1) put current land into q and v
                (2) for each neighbor node:
                        if it is land and it has not been put into q and v
                            dfs(new node)
        
        TC: O(m * n)

		SC: O(m * n)
        """
```

### 1136. Parallel Courses

```python
class Solution:
    from collections import defaultdict, deque
    def minimumSemesters(self, n: int, edges: List[List[int]]) -> int:
        graph = defaultdict(list)
        degree = [0] * n
        q = deque()
        v = set()
        
        #initialize graph and degree with edges: [a, b] : a --> b
        for a, b in edges:
            degree[b - 1] += 1
            graph[a - 1] += [b - 1]
        
        #initialize q and v
        for i in range(n):
            if degree[i] == 0:
                q += [i]
                v.add(i)
                
        level = 1
        while q:
            size = len(q)
            for _ in range(size):
                i = q.popleft()
                for j in graph[i]:
                    if degree[j] >= 1:
                        degree[j] -= 1
                        if degree[j] == 0 and j not in v:
                            q += [j]
                            v.add(j)
            level += 1
        #Notice that cost will + 1 when we pop out node of last level
        if len(v) == n:
            return level - 1
        else:
            return -1
        """
        Data Structure:
            graph[i] : list of next courses of i
            degree[i]: num of previous courses of i
            q: all courses of the same level
            v: visited courses
        
        Algorithm:
            1.  initialize graph and degree
            
            2.  push all courses whose degree is 0 into q and v
            
            3.  while q:
                    (1) pop out all courses of the same level
                    (2) for next courese of graph[current course]:
                            degree[next] -= 1
                            if degree[next] == 0 and unvisited:
                                add it to q
                                add it to v
                    (3) level += 1
            4.  if len(v) == n, return level
                else:   -1
        
        SC: O(V + E)
        TC: O(V + E)
        """
```



### 1494. Parallel Courses II

```python
from itertools import combinations
class Solution:
    def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
        g = defaultdict(list)
        d = [0] * n
        
        for a, b in relations:
            g[a - 1] += [b - 1]
            d[b - 1] += 1
        
        @lru_cache(None)
        def dfs(state, d):
            if state == (1 << n) - 1:
                return 0
            
            take = []
            for i in range(n):
                if (1 << i) & state == 0 and d[i] == 0:
                    take += [i]
            
            res = inf
            for cs in combinations(take, min(k, len(take))):
                nstate = state
                nd = [n for n in d]
                for c in cs:
                    nstate |= (1 << c)
                    for nc in g[c]:
                        nd[nc] -= 1
                
                res = min(res, 1 + dfs(nstate, tuple(nd)))
                for c in cs:    
                    nstate ^= (1 << c)
                    for nc in g[c]:
                        nd[nc] += 1
            return res
                
        return dfs(0, tuple(d))
        """
        Not BFS
        
        DP + DFS
        
        Explanation:
        
        	First of all, we still need g and d
        
        	In addition, we need state (taken or not taken) of each course to choose which course to take
        	
        	(1) add current course to state
        	(2) degree of all next course -= 1
        
        Data Structure:
            1.  dfs(mask, degree): return min semester to take all left course
                (1) mask: which course is available
                    0 is available, 1 is unavailble
                (2) degree: degree of current courses
            
            2.  graph[i]: list of next courses of course i
            
        Algorithm:  
            1. initialize degree and graph
            
            2.  dfs(mask, degree):
                1.  mask == (1 << n) - 1:  return 0
                
                #if dp[mask] is not None:
                #    return dp[(mask, degree)]
                2.  (1) initialize a list to store all availble courses
                        <1> 0 in state
                        <2> degree[i] == 0
        
                    (2) pick up k courses (or len(list))
                       		a. for each courses in k picked courses:
                               <1> update mask --> set it as unavailble
                               <2> update degree --> degree[next course] -= 1
									temp = 1 + dfs(new mask, new degree)
                            b. res = min(res, temp)
                                    
                            c.  for each courses in k picked courses:
                                    	convert it back

                   	(3) return res
                       	#dp[(mask, degree)] = res
                    
        Tuple:  hashable but cann't be assigned a val --> d[a] = b X if d is a tuple
        
        List:   unhashable but can be assigned a val --> d[a] = b X if a is a list(d is a {})
        
        SC: max depth of recursion tree + d --> O(n)
        TC: number of states * c n k * k --> O(2 ^ n * c n k * k)
        """
```



### 1284. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix

```python
class Solution:
    def minFlips(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        start = 0
        for i in range(m):
            for j in range(n):
                num = mat[i][j]
                start |= num << (i * n + j)
        
        q = collections.deque()
        v = set()
        
        q += [start]
        v.add(start)
        level = 0
        target = 0
        
        while q:
            size = len(q)
            for _ in range(size):
                s = q.popleft()
                if s == target:
                    return level
                for i in range(m):
                    for j in range(n):
                        ns = s
                        for di, dj in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < m and 0 <= nj < n:
                                ns ^= 1 << (ni * n + nj)
                        if ns not in v:
                            q += [ns]
                            v.add(ns)
            level += 1
        return -1
        
        """
        Explanation:
            row i, col j in mat--> i * n + j in string
        
        Data Structure:
            q:  string of mat of same level
            v:  visited string of mat
        
        Algorithm:
            1.  initialize start string and put it in q and v
            
            2.  pop out all strings of same level:
                (1) if string == target: return level
                (2) for each pos:
                        <1> for cur and neighbor cell:
                            if valid:
                                convert it
                        <2> if new s unvisited:
                            add it to q and v
                            
        SC: size of q :O(2 ^ mn)
        TC: O( mn * 2 ^ mn)
        """
```



### 1197. Minimum Knight Moves

```python
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        q = collections.deque()
        v = set()
        q += [(0, 0)]
        v.add((0, 0))
        level = 0
        while q:
            size = len(q)
            for _ in range(size):
                i, j = q.popleft()
                if i == x and j == y:
                    return level
                for di, dj in [(1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (-2, 1), (2, -1), (-2, -1)]:
                    ni, nj = i + di, j + dj
                    if (ni, nj) not in v:
                        q += [(ni, nj)]
                        v.add((ni, nj))
            level += 1
        return -1
        
        """
        Data Structure:
            q: all pos of same level
            v: visited pos
            
        Algorithm:
            1.  initialize q and v with (0, 0)
            
            2.  (1) pop out all nodes of the same level
                
                (2) if cur i, j == target, return level
                
                (3) for each new ni, nj:
                        if valid and not in v, add it to q and v
        
        TC:O(|x| * |y|)
        SC:O(|x| * |y|)
        """
```





### 1284. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix

```python
class Solution:
    def minFlips(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        start = 0
        for i in range(m):
            for j in range(n):
                num = mat[i][j]
                start |= num << (i * n + j)
        
        q = collections.deque()
        v = set()
        
        q += [start]
        v.add(start)
        level = 0
        target = 0
        
        while q:
            size = len(q)
            for _ in range(size):
                s = q.popleft()
                if s == target:
                    return level
                for i in range(m):
                    for j in range(n):
                        ns = s
                        for di, dj in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < m and 0 <= nj < n:
                                ns ^= 1 << (ni * n + nj)
                        if ns not in v:
                            q += [ns]
                            v.add(ns)
            level += 1
        return -1
        
        """
        Explanation:
            start to cut from non-apple leaves
        
        	if next node is apple,  we cut edge but do not push it into queue
        
        Data Structure:
            q:  string of mat of same level
            v:  visited string of mat
        
        Algorithm:
            1.  initialize start string and put it in q and v
            
            2.  pop out all strings of same level:
                (1) if string == target: return level
                (2) for each pos:
                        <1> for cur and neighbor cell:
                            if valid:
                                convert it
                        <2> if new s unvisited:
                            add it to q and v
        """
```

TC: O(O(2^(m*n)))

SC: O(O(2^(m*n)))

### 1443. Minimum Time to Collect All Apples in a Tree

```python
class Solution:
    from collections import deque, defaultdict
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        hasApple[0] = True
        g = defaultdict(list)
        d = [0] * n
        q = deque()
        v = set()
        
        for i, j in edges:
            d[i] += 1; d[j] += 1
            g[i] += [j]
            g[j] += [i]
        
        for i in range(n):
            if d[i] == 1 and not hasApple[i]:
                q += [i]
                v.add(i)
        
        while q:
            size = len(q)
            for _ in range(size):
                i = q.popleft()
                for j in g[i]:
                    if d[j] >= 1 and d[i] -= 1:
                        d[j] -= 1; d[i] -= 1
                        if d[j] == 1 and (not hasApple[j]) and (j not in v):
                            q += [j]
                            v.add(j)
        return sum(d)
        """
        Course Schedule
        
        Explanation:
            res = degree of all nodes of apple and root of apple
            -->
            remove all degree of nodes of children of apple
            
        Data Structure:
            q: all non apple nodes of same level'
            
            v: visited nodes
            
            g: next nodes
            
            d: degree of each node
            
        Algorithm:
            1.  initialize graph and degree
                initialize hasApple[root] = True: we only delete a subtree and we cannot delete neighbot tree
            
            2.  add all non apple leaves to q
            
            3.  while q:
                (1) pop out all nodes of same level
                (2) for each next node:
                        if degree of next ndoe > 0
                        <1> degree of next node -= 1
                        <2> if degree == 1 and not apple:
                                add it to queue and v
            
            4.  return sum of degree
        
        TC: O(V + E)
        SC: O(V + E)
        """
```

TC: O(n)

SC: O(n)



## Tree

### 98. Validate Binary Search Tree

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node, lb, rb):
            if node == None:
                return True
            if node.val < lb or node.val > rb:
                return False
            return dfs(node.left, lb, node.val) and dfs(node.right, node.val, rb)
        return dfs(root, -inf, inf)
        """
        Data Structure:
            def dfs(left boundary and right boundary):
                (1) root in Given left boundary and right boundary
                (2) left subtree and right subtree ok or not?
                
        Algorithm:
            1.  p --> r
                    if left bounrdary <= val <= right boundary:
                        ok, check  r-- >c
                    else
                        false
            
            2.  r --> c
                    dfs(root.left, update right boundary = root.val)
                    dfs(root.right, update left boundary = root.val)
        TC:O(n)
        SC:O(n)
        """
```

### 99. Recover Binary Search Tree

```python
class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
    
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        prev = TreeNode(-inf)
        first, second = None, None
        def dfs(node):
            nonlocal prev, first, second
            if not node:
                return
            dfs(node.left)
            if not first and prev.val > node.val:
                first = prev
            if prev.val > node.val:
                second = node
            prev = node
            dfs(node.right)
        
        dfs(root)
        if first and second:
            first.val, second.val = second.val, first.val
        """   
        Explanation:
            (1)
            1,      2,      3,      4,      5
            -->swap 2 and 4
            1,      4,      3,      2,      5
           <1>      prev    cur
           <2>              prev    cur
           
           <1>      first           
           <1>         		second
           <3>         				second
                    
            (2)
            1,      2,      3,      4,      5
            -->swap 2 and 3
            1,      3,      2,      4,      5
                    prev    cur
                    prev    cur
                    first   
                    		second
            
        Data Structure:
            1. prev
            2. cur
            
        Algorithm:
            1. iterate nodes with inorder sort
                (1): if first not found and prev > cur
                        set prev as first
                (2): if first found and prev > cur
                        set cur as second
                        
            3. swap value of first and second
            
        TC:O(n)
        SC:O(n)    
        """
        
```

### 108. Convert Sorted Array to Binary Search Tree

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(l, r):
            if l > r:
                return None
            m = l + (r - l) // 2
            node = TreeNode(nums[m])
            node.left = dfs(l, m - 1)
            node.right = dfs(m + 1, r)
            return node
        return dfs(0, len(nums) - 1)
        """
        Data Structure:
            left, right
            
        Algorithm:
            set mid as root
            root.left = left ~ mid - 1
            root.right = = dfs(mid + 1, right)
        
        TC:O(n)
        SC:O(n)
        """
```

### 109. Convert Sorted List to Binary Search Tree

```python
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
        
class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        
        def dfs(head):
            if head is None:
                return None
            if head.next is None:
                return TreeNode(head.val)
            
            prev, slow, fast = None, head, head
            while fast and fast.next:
                prev, slow, fast = slow, slow.next, fast.next.next
            
            node = TreeNode(slow.val)
            prev.next = None
            node.left = dfs(head)
            node.right = dfs(slow.next)
            return node
        
        return dfs(head)
        
        """
        Data Structure:
            dfs(head): return root of BST constrcuted by head
            
        Algorithm:
            dfs:
            1.  if head is None:
                    return None
                
                if head.next is None:
                    return TreeNode(head.val)
                
                
            2.  choose mid node and prev(last node of mid node)
        
            3.  prev.next = None
                root = mid
                root.left = dfs(head)
                root.right = dfs(mid.next)
                
        TC:O(n)
        SC:O(n)        
        """
```

### 114. Flatten Binary Tree to Linked List

```python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if not root:
            return
        
        self.flatten(root.left)
        self.flatten(root.right)
        
        l, r = root.left, root.right
        
        root.left = None
        root.right = l
        
        while root.right:
            root = root.right
            
        root.right = r
        """
        Data Structure:
            dfs(node): turn tree rooting at node into linkedlist
        
        Algorithm:
            1. flatten left tree and right tree
            
            2.  node.left = None
                node.right = flattened left tree
                
            3. move to right most child node
            
            4. node.right = flattened right tree
            
        TC:O(n)
        SC:O(n)    
        """
```

### 124. Binary Tree Max Path Sum

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = -inf
        
        def dfs(node):
            if node is None:
                return 0
            
            l = max(dfs(node.left), 0)
            r = max(dfs(node.right), 0)
            
            self.res = max(self.res, l + r + node.val)
            
            return max(l, r) + node.val
        
        dfs(root)
        return self.res
        """
        Data Structure:
            dfs(node):
                (1) update res with node connector
                (2) return max path rooting at node
        
        Algorithm:
            1.  r -- > c
                update res and return max path sum with dfs(root.left)
                update res and return max path sum with dfs(root.right)
                
            2.  c --> r
                if max path sum of left < 0 --> left_sum = 0
                if max path sum of right < 0 --> rigth_sum = 0
                
                total = root.val + left_sum + right_sum
                update res with total
            
            3.  r --> p
                return root.val + max(left_sum, rigth_sum)
        
        TC:O(n)
        SC:O(n)
        """
```

### 236. Lowest Common Ancestor of a Binary Tree

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        
        if root.val == p.val or root.val == q.val:
            return root
        
        l, r = self.lowestCommonAncestor(root.left, p, q), self.lowestCommonAncestor(root.right, p, q)
        
        if l and r:
            return root
        if not l and not r:
            return None
        if l or r:
            return l if l else r
        """
        Data Structure:
            dfs --> return (1) p or q or (2) node of other val (3) None
            p / q : (1) p / q is lowest ancestor (2) p / q is found
            node of other val: node is loweset ancestor
        
        Algorithm:
            (1) if current node is p / q:
                    return p / q
            
            (2) find p / q / lowest ancestor in left / right subtree
                    l, r = dfs(root.left), dfs(root.right)
                    
                <1> if l and r are not None --> p and q are found in different subtrees:
                        root is loweset ancestor
                        return root
                <2> if l or r is None
                        return the not None one (p / q / node of other val)
                <3> if l and r are both None:
                        return None
        
        TC:O(n)
        SC:O(n)
        """
```

### 250. Count Univalue Subtrees

```python
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(node):
            if node is None:
                return True
            left = dfs(node.left) and (node.left is None or node.val == node.left.val)
            right = dfs(node.right) and (node.right is None or node.val == node.right.val)
            
            if left and right:
                self.res += 1
                return True
            return False
        dfs(root)
        return self.res
        
        """
        Data Structure:
            dfs(node): 
            (1) return univalue or not / empty or not
            (2) update res with root is unival or not
            
        Algorithm:
            c -- >r
                (1) dfs(left / right) : left / right subtree is univalue or empty
                (2) root == left or left is empty
        
        TC:O(n)
        SC:O(n)
        """
```



### 617. Merge Two Binary Trees

```python
class Solution:
    def mergeTrees(self, r1: Optional[TreeNode], r2: Optional[TreeNode]) -> Optional[TreeNode]:
        if r1 is None and r2 is None:
            return None
        if r1 is None or r2 is None:
            return r1 if r2 is None else r2
        cur = TreeNode(r1.val + r2.val)
        cur.left = self.mergeTrees(r1.left, r2.left)
        cur.right = self.mergeTrees(r1.right, r2.right)
        
        return cur
    
    """
    Data Structure:
        1. dfs(node1, node2): return a combined tree
        
        2.  node
            node.left
            node.right
    
    Algorithm:
        1.  if both node1 and node2 are None:
            return None
        
        2.  if one of node1 and node2 is None:
            return non-None one
        
        3.  if both node1 and node2 are not None: 
            new node.val = node1.val + node2.val
            new node.left = dfs(node1.left, node2.left)
            new node.right = dfs(node1.right, ndoe2.right)
    
    TC:O(n)
    SC:O(n)
    """
```



### 965. Univalued Binary Tree

```python
class Solution:
    def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        if self.isUnivalTree(root.left) and (root.left is None or root.val == root.left.val) and self.isUnivalTree(root.right) and (root.right is None or root.val == root.right.val):
            return True
        return False
        
    """
    Data Structure:
        1.  isUnivalueTree(root): True or False
    
    Algorithm:
        1.  if root is None:
                return True
        2.  res = isUnivalueTree(root.left) and (left is None or left.val == root.val)
                    and right...
                    
    TC:O(n)
    SC:O(n)              
    """
```



### 1552. Diameter of N-Ary Tree

```python
class Solution:
    def diameter(self, root: 'Node') -> int:
        res = 0
        def dfs(node):
            nonlocal res
            first = second = 0
            for c in node.children:
                d = dfs(c)
                if d > first:
                    first, second = d, first
                elif d > second:
                    second = d
            res = max(res, first + second) #Notice dismater is number of edges
            return first + 1
        dfs(root)
        return res
    
    """
    Explanation:
        how to get first and second biggest num in an array
            if num > first:
                update first, second = cur, first
            elif num > second:
                update second = cur
        
        how to get k biggest num in an array:
            min heap
    
    Data Structure:
        1.  dfs(node): return longest path with root node
        
        2.  first, second longest path for each dfs
        
    
    Algorithm:
        dfs(node)
        1.  for each child:
                cur = dfs(child)
                if num > first:
                    update first and second
                elif num > second:
                    only update second
            
        2.  update res with first + second
        
        3.  return 1 + first
        
    TC:O(n)
    SC:O(n) 
    """
```



### 1660. Correct a Binary Tree

```python
class Solution:
    def correctBinaryTree(self, root: TreeNode) -> TreeNode:
        seen = set()
        
        def dfs(root):
            if not root or (root.right and root.right.val in seen):
                return
            seen.add(root.val)
            root.right = dfs(root.right)
            root.left = dfs(root.left)
            return root
            
        return dfs(root)
    
    """
    Explanation:
    	there will be a problem if current node points to a visited right children or node of higher level
        -->
        (1) current node should not point to parent
        (2) if current node is in left subtree of a node, it should be point to any node in right subtree
        
        -->
        we need to iterate and store right children and parents.
        cur --> right --> left

    Data Structure:
        1.  seen: previous parents and nodes in right subtree
        
        2.  dfs(node): preOrder
            (1) check if current node can be deleted
            (2) dfs(node.right) and dfs(node.left) to clean subtree
            (3) return a cleaned subtree
        
    Algorithm:
        dfs(node):
        (1) if node is empty or node.right in seen:
                return None
        (2) seed.add(node)
        (3) node.right = dfs(node.right)
            node.left = dfs(node.left)
        (4) return node
    """
```



## Design

### 146. LRU Cache

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.d = collections.OrderedDict()
        self.k = capacity

    def get(self, key: int) -> int:
        if key in self.d:
            val = self.d[key]
            del self.d[key]
            self.d[key] = val
            return self.d[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            del self.d[key]
            self.d[key] = value
        else:
            if len(self.d) == self.k:
                self.d.popitem(last = False)
            self.d[key] = value
        return
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
    """
    Explanation:
        cache --> [d] capacity k --> only k [key, val]
        LRU cache   --> keys are sorted by insertion time

        cache --> d
        LRU cache --> orderedict
    
    
    Data Structure:
        OrderedDict: d
        capacity: k
        
    Algorithm:
        def get(key): (we do not need to consider capacity here)
            1. if key not exists in d:
                    return -1
            2. else if it does exist:
                    (1) delete item [key, val]
                    (2) put [key, val] into d again
                    (3) return val
            
        def put(key, val):
            1. if key exists in d:
                    (1) delete original [key, old val]
                    (2) put back [key, new val]
            2.  else key does not exist:
                    (1) if size == capacity:
                            pop out the least recenlty used item --> popitem(last = False)
                            put [key, new val]
                    (2) else:
                            put [key, new val]
    """
```

### 208. Implement Trie

```python
class Trie:
    def __init__(self):
        self.d = collections.defaultdict(Trie)
        self.isWord = False

    def insert(self, word: str) -> None:
        cur = self
        for c in word:
            cur = cur.d[c]
        cur.isWord = True
        
    def search(self, word: str) -> bool:
        cur = self
        for c in word:
            if c not in cur.d:
                return False
            else:
                cur = cur.d[c]
        return cur.isWord
        

    def startsWith(self, prefix: str) -> bool:
        cur = self
        for c in prefix:
            if c not in cur.d:
                return False
            else:
                cur = cur.d[c]
        return True

    """
    Explanation:
    "word": root/self - w - o - r - d(isWord)
    "world"                   - l - d(isWord)
    
    Data Structure:
        Trie:
            d = {char, Trie(children)}
            boolean isWord
    
    Algorithm:     
    1.  insert(word)
        (1)for each char in word:
                create Trie Node with char & get into it
                --> cur = cur.d[c]
        (2)set last char as word:
        
    2.  search(word)
        (1)for each char in word:
                if not exist in d(children) --> False
        (2)if last char is word: return True
        
    3.  startsWith(prefix)
        (1)for each char in word:
                if not exist in d(children) --> False
        (2)if make true going down to last: return True
        
    L: length of key
    TC:O(L)
    SC:O(L)
    """
```



### 1804. Implement Trie II (Prefix Tree)

```python
class Trie:
    def __init__(self):
        self.d = collections.defaultdict(Trie)
        self.w = 0
        self.p = 0

    def insert(self, word: str) -> None:
        cur = self
        for c in word:
            cur = cur.d[c]
            cur.p += 1
        cur.w += 1

    def countWordsEqualTo(self, word: str) -> int:
        cur = self
        for c in word:
            if c not in cur.d:
                return 0
            cur = cur.d[c]
        return cur.w

    def countWordsStartingWith(self, prefix: str) -> int:
        cur = self
        for c in prefix:
            if c not in cur.d:
                return 0
            cur = cur.d[c]
        return cur.p

    def erase(self, word: str) -> None:
        cur = self
        for c in word:
            cur = cur.d[c]
            cur.p -= 1
        cur.w -= 1


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.countWordsEqualTo(word)
# param_3 = obj.countWordsStartingWith(prefix)
# obj.erase(word)
    """
    Data Structure:
        TrieNode:
        1.children(next char)
        2.num of word
        3.num of prefix

        Trie:
        1. root of TrieTree (empty node without corresponding char)

        2. 3 functions
    
    Algorithm:
    1.insert
        (1) start from root of TrieTree --> node = self.root
        (2) for each c:
                1> put c in children of node
                2> update num of prefix
                3> node = c
                
        (3) for last c:
                update num of word
                
    2.count words
        (1) start from root of TrieTree --> node = self.root
        (2) for each c:
                if not exist in children --> return 0
                else delve deeper
            return num of word
        
    3.count prefix
        (1) start from root of TrieTree --> node = self.root
        (2) for each c:
                if not exist in children --> return 0
                else delve deeper
            return num of prefix
            
    4.erase(guarantee existence of word)
        (1) start from root of TrieTree --> node = self.root
        (2) for each c:
                update num of word
        (3) for last c: 
                update num of prefix
    
    TC:O(L)
    SC:O(L)
    """
```



### 239. Sliding Window Maximum

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        res = []
        for i, n in enumerate(nums):
            while dq and nums[dq[-1]] < n:
                dq.pop()
            while dq and dq[0] <= i - k:
                dq.popleft()
            dq += [i]
            if i >= k - 1:
                res += [nums[dq[0]]]
        return res
        """
        Data Structure:
            1.  deque()
                dq[0] [c1, c2, c3 ... cn] dq[-1]
                dq stores all candidates for max num of current num i
                -->
                before adding numn into deque
                we need to pop all smaller candidates
                and all candidates whose index < n - k + 1
                ...
                num1 > num2 > num3 > ...
                
        Algorithm:
            1.  if length from dq[0] to i > k:
                    dq.popleft()
                    
            2.  while dq[-1] < current num:
                    dq.pop()
                    
            3.  add current num to dq
            
            4.  res += [dq[0]]
        """
```



### 295. Find Median from Data Stream

```python
class MedianFinder:
    from heapq import heappop, heappush
    def __init__(self):
        self.bigger = []
        self.smaller = []

    def addNum(self, num: int) -> None:
        if len(self.bigger) == len(self.smaller):
            heappush(self.smaller, -num)
            temp = -heappop(self.smaller)
            heappush(self.bigger, temp)
        else:
            heappush(self.bigger, num)
            temp = heappop(self.bigger)
            heappush(self.smaller, -temp)

    def findMedian(self) -> float:
        if len(self.bigger) == len(self.smaller):
            return (self.bigger[0] - self.smaller[0]) / 2        
        else:
            return self.bigger[0]

    """
    Data Structure:
        1. bigger, min heap: store biggest one
        
        2. smaller, max heap: store smallest one
        
    Algorithm:
        1. addNum:
            if len of bigger == len of smaller, we need to add num to bigger:
                (1) add new num to smaller
                (2) pop out biggest num in smaller
                (3) add biggest num to bigger
            elif len of bigger > len of smaller, we need to add num to smaller：
                (1) add new num to bigger
                (2) pop out smallest num in bigger
                (3) add smallest num to smaller
        
        2. findMedian:
            (1) if len of bigger + len of smaller is odd:
                    return smallest of bigger
                    
            (2) if len of bigger + len of smaller is even:
                    return biggest of smaller + smallest of bigger / 2
    
    TC: addNum : O(logn) --> n: number of nums
    SC: O(n)
    """
```

### 480. Sliding Window Median

```python
class Solution:
    from heapq import heappush, heappop
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        smaller, bigger = [], []
        res = []
        
        def move(h1, h2):
            num, idx = heappop(h1)
            heappush(h2, (-num, idx))
            
        def getMedian(h1, h2, k):
            return h2[0][0] * 1.0 if k & 1 else (h2[0][0] - h1[0][0]) / 2
        
        for i in range(k):
            heappush(bigger, (nums[i], i))
        
        for _ in range(k // 2):
            move(bigger, smaller)
            
        res += [getMedian(smaller, bigger, k)]
            
        for l_idx, r in enumerate(nums[k:]):
            l = nums[l_idx]
            r_idx = l_idx + k
            if r >= bigger[0][0]:
                heappush(bigger, (r, r_idx))
                if not smaller or l <= -smaller[0][0]: # need to check empty
                    move(bigger, smaller)
            else:
                heappush(smaller, (-r, r_idx))
                if l >= bigger[0][0]: # no need to check empty
                    move(smaller, bigger)
            
            while smaller and smaller[0][1] <= l_idx:
                heappop(smaller)
            while bigger and bigger[0][1] <= l_idx:
                heappop(bigger)
            
            res += [getMedian(smaller, bigger, k)]
        
        return res
    """
    Explanation:
        For lazy removing, we only pop elements that may influence the final result
    
    Data Structure:
        1.  small and large: smaller and larger elements (num, index) in sliding window
            (1) all values in smaller < values in larger
            (2) num of valid ele in smaller (+ 1) == num of valid ele in larger
        
        2.  def move(h1, h2)]
            move top ele from h1 to h2
            
        3.  def median
            get median of current sliding window with top element in smaller and larger
        
        
    Algorithm:
        1.  initialize smaller and larger:
            (1) push first k elements into larger
            (2) move top k // 2 elements into smaller
            (3) add cur median to res
            
        2.  for each new end of sliding window:
        
            smaller[0] may be equal to larger[0]
            if new end > large[0], add it to large
            elif new end < small[0], add it to small
            else new end == large[0] or new end == small[0], add it to large[0] or small[0]
        
            
            (1) add end to larger if new val > smallest val in larger (else add it to smaller)
            (2) Assume old start of sliding has been lazily removed
                if old start is in larger: old start > large[0][0]
                    num of ele valid vals is ok
                else(smaller is empty or old start is in smaller) not smaller or new end <= biggest num in smaller
                    num of valid ele in smaller < needed
                    num of valid ele in larger > needed
                    -->
                    move one ele from larger to smaller
        
        3.  pop out all elements in smaller and larger that can affect our fianl result:
                if index of top element of smaller/larger <= index of old start
        
        4.  add current median to res
        
        
    TC: insertion of each element : O(logn),  total number of removal: O(n - k)
    	O(nlogn)
    
    SC: O(n) two heaps
    
    """
```



### 1825. Finding MK Average

```python
from sortedcontainers import SortedList
from collections import deque
class MKAverage:
    def __init__(self, m: int, k: int):
        self.m, self.k = m, k
        self.q = deque()
        self.sl = SortedList()
        self.total = self.first_k = self.last_k = 0

    def addElement(self, num: int) -> None:
        self.total += num
        self.q += [num]
        i = self.sl.bisect_left(num)
        if i - 0 + 1 <= self.k:
            self.first_k += num
            if len(self.sl) >= self.k:
                self.first_k -= self.sl[self.k - 1]
        
        if len(self.sl) - i + 1 <= self.k:
            self.last_k += num
            if len(self.sl) >= self.k:
                self.last_k -= self.sl[-self.k]
                
        self.sl.add(num)
        
        if len(self.sl) == self.m + 1:
            num = self.q.popleft()
            i = self.sl.bisect_left(num)
            self.total -= num
            if 0 <= i <= self.k - 1:
                self.first_k -= num
                self.first_k += self.sl[self.k]
                
            elif len(self.sl) - self.k <= i <= len(self.sl) - 1:
                self.last_k -= num
                self.last_k += self.sl[len(self.sl) - self.k - 1]
        
            self.sl.remove(num)
        
    def calculateMKAverage(self) -> int:
        if len(self.sl) < self.m:
            return - 1
        else:
            return (self.total - self.first_k - self.last_k) // (self.m - 2 * self.k) 

"""
    1756 - Design most Rencently Used Queue
    SortedList
    Explanation:
        1.  Add
            before adding cur num to sortedList, we check its insertion index i
            Assume we have inserted it into sortedlist(but in fact we have not):
                all left numbers is from 0 to i:
                    if length <= k, 
                        we need to add current num to first_k 
                        and delete original last k if length of SortedList has reached k
                    
                all right numbers including itself is from i to len(list)
                    if length <= k
                        we need to add current num to last_k 
                        and delete original last k if length of SortedList has reached k
                        
            
        
        2. Remove
            Now we have added num to sortedList, if its length == m + 1, we need to remove first added num
                
            find index of first num:
                if 0 <= index <= k - 1 --> remove it from first_k and add a new num to first_k 
                
                elif n - k <= index <= n - 1 --> remove it from last_k and adda num to last_k
                
                else: delete it from sortedList directly
        
    
    Data structure
    1. deque q 
        sort of being added
    
    2. sortedList sl
        sort of val
        
    3. sum, sum of first k, sum of last k of sortedList
    
    Algorithm:
    1.  compute:
        res = sum - sum of first k - sum of last k / m - 2 * k
        
    2.  Add
        (1) add to deque q and update total
        (2) update first_k and last_k by num to add
                <1> calculate insertion index of num --> at most n
        
                <2> if after adding, there are at most k numbers from 0 to i: i - 0  + 1 <= k --> i <= k - 1
                    --1 add it to first_k
                    --2 delete num with index k - 1 in original SortedList
                
                <3> if after adding,there are at most k numbers from i to n: n - i + 1 <= k --> i >= n - k + 1
                    --1 add it to last_k
                    --2 delete num with index n - k in original SortedList
                    
         (3)	add num to sortedList
                
        Remove if len(sortedList) == m + 1
        (1) remove from deque and update total
        
        (2) update first_k and last_k by num to delete
               	<1> calculate index of num by index at most n - 1
               	
                <2> if num in [0, k - 1]: remove it from first_k and add [k] to first_k
                    if num in [-k, -1]: remove it from last_k and add[-k - 1] to last_k
                    
        (3) remove it from sortedList
    
    SC: size or SortedList and size of q --> O(m)
    
    TC: add num --> O(logm)
        calculate avg --> O(1)
    """
```



### 705. Design HashSet

```python
class MyHashSet:
    def __init__(self):
        self.size = 10000
        self.buckets = [[] for _ in range(self.size)] 
        #Notice that we can only use for _ in range to duplicate list into list of list

    def add(self, key):
        bucket, i = self.find(key)
        if i >= 0:
            return
        bucket += [key]

    def remove(self, key):
        bucket, i = self.find(key)
        if i < 0:
            return
        bucket.remove(key)

    def contains(self, key):
        bucket, i = self.find(key)
        return i >= 0

    def find(self, key):
        idx = key % self.size
        bucket = self.buckets[idx]
        for i, k in enumerate(bucket):
            if k == key:
                return bucket, i
        return bucket, -1

# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)

    """
    Data Structure:
    size of bucket list in bucket
    
    bucket list[bucket1, bucket2, ...]
    num list in bucket [num1, num2,...]
    
    Algorithm:
    1 def hash(key):
        (1) index of bucket = num % size
        (2) find index of key in bucket by enumeration
                if found: return index of bucket, index of key
                else: return index of bucket, index of key == -1
    
    2 def add
        (1) get index of bucket and key
        (2) if index of key != -1: end
            else bucket.append(key)
    
    3 def remove
        (1) get index of bucket and key
        (2)  if index of key == -1: end
             else bucket.remove(key)
    
    4 def contains
        (1) get index of bucket and key
        (2) return key != -1
        
    5.def find(key)
        return bucket in which key should be and (idx of key or -1)
        
    TC: O(K) find --> number of insertion keys, all of them are inserted into one bucket
    SC: O(H + K) size of hashmap H(number of buckets) + number of insertion keys K
    """
```



### 706. Design HashMap

```python
class MyHashMap:
    def __init__(self):
        self.size = 10000
        self.buckets = [[] for _ in range(self.size)]

    def put(self, key: int, value: int) -> None:
        bucket, i = self.find(key)
        if i >= 0:
            bucket[i] = (key, value)
        else:
            bucket += [(key, value)]

    def get(self, key) -> int:
        bucket, idx = self.find(key)
        if idx == -1:
            return -1
        else:
            return bucket[idx][1]

    def remove(self, key: int) -> None:
        bucket, idx = self.find(key)
        if idx != -1:
            bucket.remove(bucket[idx])
        else:
            return
        
    def find(self, key):
        idx_bucket = key % self.size
        bucket = self.buckets[idx_bucket]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                return bucket, i
        return bucket, -1


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```



## Heap

### 23. Merge K Sorted Lists

```python
class Solution:
    from heapq import heappush, heappop
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        res = cur = ListNode(0)
        
        h = []
        for i, n in enumerate(lists):
            if n:
                heappush(h, (n.val, i, n))
        
        while h:
            v, i, n = heappop(h)
            
            cur.next = n
            cur = cur.next
            
            if n.next is None:
                continue
            else:
                heappush(h, (n.next.val, i, n.next))
        
        return res.next
        """
        Data Structure:
            heap --> store k pointers --> (val, num of list, node)
            i is to debug when val is equal
            
        Algorithm:
            1 initialize heap:
                    for each list, store its head (val, i ,node)
            
            2 res = cur = ListNode(0)
                while h:
                    (1) pop out samllest node and add it to cur 
                    
                    (2) update heap:
                        <1> if cur node is last: just pop
                        <2> if cur node is not last: pop and push
                    v, i, node = h[0]
                    
                    temp = ListNode(v)
                    cur.next =temp
                    cur = temp
                    
                    (1) node.next is None:
                            heappop(h) --> delete h[0]
                    (2) delete h[0]
                        add node.next(v, i, node) to heap
            
            3.  return res.next
            
        SC: O(k) size of heap --> number of lists
        TC: O(Nlogk) N: number of nodes
        """				
```



### 373.Find K Pairs with Smallest Sums

```python
class Solution:
    from heapq import heappush, heappop
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        h = []
        v = set()
        m, n = len(nums1), len(nums2)
        
        heappush(h, (nums1[0] + nums2[0], 0, 0))
        v.add((0, 0))
        
        res = []
        while h and k > 0:
            _, i, j = heappop(h)
            res += [(nums1[i], nums2[j])]
            for di, dj in [(1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if ni < m and nj < n and (ni, nj) not in v:
                    heappush(h, (nums1[ni] + nums2[nj], ni, nj))
                    v.add((ni, nj))
            k -= 1
        return res
        
        """
        Data Structure:
            h: min heap, store(v, i, j) --> i is index in nums1, j is index in nums2
            v: store(i, j) in heap
            
        Algorithm:
            1. initialize heap:
                add nums1[0] and nums2[0] to heap
                
            2. while h :
                (1) pop the smallest one, k -= 1
                (2) next dir: 1, 0 or 0, 1
                        if in boundary and not vistied:
                            add it to heap
        
        SC: v + h: O(m * n) m:len of nums1, n: len of num2, there are m * n pairs in total
        TC: h: k *log (mn)
        """
```



### 1439. Find the Kth Smallest Sum of a Matrix With Sorted Rows

```python
from heapq import heappush, heappop
class Solution:
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        h = []
        v = set()
        m, n = len(mat), len(mat[0])
        s1 = sum([mat[i][0] for i in range(m)])
        s2 = tuple([0] * m)
        #Notice that this is how we duplicate single num into a list of num
        heappush(h, (s1, s2))
        v.add(s2)
        res = 0
        
        while k and h:
            cur_sum, cur_cols = heappop(h)
            res = cur_sum
            for row in range(m):
                n_sum = cur_sum
                n_cols = [col for col in cur_cols]	#Notice that we need to copy cols first
                cur_col = n_cols[row]
                n_cols[row]  = cur_col + 1
                if tuple(n_cols) not in v and n_cols[row] <= n - 1:
                    n_sum = n_sum - mat[row][cur_col] + mat[row][cur_col + 1]
                    heappush(h, (n_sum, n_cols))
                    v.add(tuple(n_cols))
            k -= 1
        
        return res
        """
        Data Structure:
            heap: min heap [sum, tuple of col index from 0 to m]
            v: hashset of tuple of col index
            
        Algorithm:
            1.  initialize heap = [(sum of mat[i][0], (0, ... 0))]
                v = set((0, ... ,0))
            
            2.  while k > 0
                (1) pop out element with min sum
                    sum = h[0][0]
                    is = h[0][1]
                    
                    res = sum    
                    
                (2) for each col index i in is:
                        <1> set a new list of index and a new sum
                        <2> i += 1
                            if new tuple of index not in v and i <= n - 1
                                new sum -= cur num and += next num
                                add (new sum and new cols) to v, h
                
                (3) k -= 1
        
        SC: v, h : O(n ^ m), number of combinations
        
        TC: h: pop out k times, and each time iterate all m rows : 
            O(m * k * log (n ^ k))
        
        """
        
```



### 347. Top K Frequent Elements

```python
class Solution:
    from heapq import heappush, heappop
    
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = collections.Counter(nums)
        
        h = []
        for n in count:
            heappush(h, (count[n], n))
            if len(h) == k + 1:
                heappop(h)
                
        res = []
        for f, n in h:
            res += [n]
        return res
    
    """
    Data Structure:
        1. count --> [key:num, val:freq]
        
        2. h: min heap [freq, num]
        
    Algorithm:
        1. get count
        
        2. for each [num, freq]:
            (1) add it to heap
            
            (2) if len(heap) > k, pop out the most freq one
            
        3.  add all nums in heap to res
    """
```



### 378. Kth Smallest Element in a Sorted Matrix

```python
class Solution:
    from heapq import heappush, heappop
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        v = set()
        (m, n) in v
        h = []
        heappush(h, (matrix[0][0], 0, 0))
        v.add((0, 0))
        
        while h and k > 1:
            _, i, j = heappop(h)
            for di, dj in [(0, 1), (1, 0)]:
                ni, nj = i + di, j + dj
                if ni < m and nj < n and (ni, nj) not in v:
                    heappush(h, (matrix[ni][nj], ni, nj))
                    v.add((ni, nj))
            k -= 1
        return h[0][0]
    
    """
    SC: O(mn)
    TC: O(klogmn)
    """
```



### 692. Top K Frequent Words

```python
class Node:
    def __init__(self, freq = 0, word = ""):
        self.freq = freq
        self.word = word
    
    def __lt__(self, other): #less-than
        if self.freq != other.freq:
            return self.freq < other.freq
        else:
            return self.word > other.word
        #define when self < other

from heapq import heappush, heappop
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        h = []
        d = collections.Counter(words)
        
        for w in d:
            f = d[w]
            node = Node(f, w)
            heappush(h, node)
            if len(h) == k + 1:
                heappop(h)
        
        res = []
        while h:
            node = heappop(h)
            res += [node.word]
        return res[::-1]
        """
        define a node: [freq, word]
        
        __lt__(self, other): less-than
        define the situation where self node < other node:
        
        a < b:  in heap, we pop out the smallest node --> least competitive one
                --> (1) min freq (2) max lexicographical order
        
        if self.freq != other.freq:
            return self.freq < other.freq
        else:
            return self.word > other.word
        
        Data Structure:
            1.  class node (freq, word)
                    node a < node b: 
                    (1) freq of a < freq of b
                    (2) word of a > word of b
            2.  heap: min heap stores node
            
        Algorithm:
            1.  get freq of each word
            
            2.  for each word and its freq:
                <1> combine them into a new node
                <2> add it into heap
                <3> if len(h) == k + 1:
                        pop out min node
        """
```



### 1046. Last Stone Weight

```python
from heapq import heappush, heappop
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        h = []
        for s in stones:
            heappush(h, -s)
            
        while len(h) > 1:
            first = -heappop(h)
            second = -heappop(h)
            heappush(h, -(first - second))
        
        return -h[0]
   	"""
   	SC: size of h O(n)
   	TC: O(nlogn)
   	"""
        
```



### 857. Min Cost to Hire K Workers

```python
from heapq import heappush, heappop
class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
        A = [(w / q, w, q) for w, q in zip(wage, quality)]
        h = [] #max heap
        total = 0
        res = inf
        for r, _, q in sorted(A):
            if len(h) == k:
                total += heappop(h)
            
            total += q
            heappush(h, -q)
            
            if len(h) == k:
                res = min(res, total * r)
        
        return res
        """
        Explanation:
            [r1 ~ rk - 1] rk:
            
            if rk is highest ratio of wage to quality, in order to get min wage
            the sum of quality from r1 to rk-1 should be minimized
        
        Data Structure:
            1. list of (ratio of wage to quality, quality) sorted by ratio from small to big
            
            2. h: max heap storing quality of worker with smaller ratio
            
            3, total: sum of quality in h
            
        Algorithm:
            1.  initialize list[ratio, quality]
            
            2.  iterate each worker by ratio:
                    (1) if enough k worker:
                        remove the person with max quality
                        from q_sum and heap
                        
                    (2) add quality of current worker to heap and total
                    
                    (3) if there are k people in h
                        update res with cur ratio * q_sum
                        
        SC: A + heap O(N + K)
        TC: sort A + heap --> O(NlogN + KlogK)
        """
```



### 1383. Max Performance of a Team

```python
from heapq import heappop, heappush
class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        mod = 10 ** 9 + 7
        h = []
        res = 0
        total = 0
        for e, s in sorted(zip(efficiency, speed), key = lambda x : -x[0]):
            if len(h) == k:
                total -= heappop(h)
                
            heappush(h, s) 
            total += s
            res = max(res, total * e)
            
        
        return res % mod
        """
        Explanation:
            1. if we choose e1 as efficiency of list, all other e2, ... en >= e1
            2. to maximize performance, choose biggest k - 1 speed
        
        Data Structure:
			1.	list of (efficiency, speed) sorted by efficiency from big to small
			2.	h: min heap storing all speeds of workers whose efficency is bigger than current worker
            3. 	sum_speed: storing sum of speed in h2
        
        Algorithm:
            pop out each efficiency in h1, iterate efficienty from max to min to set cur effiencty as min of team:
            (1) if len of h == k:
                    pop out min speed in list
                    substract it from sum
            (2) add current speed to total
                add speed to h
                
            (3) update res with cur efficiency * sum of speed
        
        """
```



## Math

### 233. Number of Digit One

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        x = 1
        res = 0
        
        while x <= n: #Notice that smaller or equal
            upperT = n // x
            T = upperT % 10
            upper = upperT // 10
            
            temp = upper * x
            
            lower = n % x
            if T == 1:
                temp += lower + 1
            elif T > 1:
                temp += x
            
            res += temp
            x *= 10
            
        return res
        """
        Explanation:
            xyz T abc:
            xyz --> upper
            abc --> lower
            (1) if upper == 000 ~ xy(z - 1):
                    current num < xyzTabc
                
                set T == 1:
                
                lower can mvoe from 000 ro 999 --> 1000
                
                --> upper * 1000
                
            (2) if upper == xyz
                    <1> T == 1:
                        lower can only move from 000 to abc:
                        --> lower + 1 
                    
                    <2> T > 1:
                        set T as 1
                        since xyz1 + lower < xyz T abc:
                        lower can move from 000 to 999
                        --> 1000 --> x
        
        Data Structure:
            x --> 当前的位数
        
        Algorithm
            1.initialze x = 1
            
            2.  while x <= n:
                (1) calculate upper, T, lower
                    upperT = n // x
                    upper = upperT // 10
                    T = upperT % 10
                    lower = n % x
                
                (2) add num of '1' of T to temp
                    temp = 0
                    temp += upper * x
                    if T == 1:
                        temp += lower + 1
                    elif T > 1:
                        temp += x
                
                (3) res += temp
                    当前的位数左移 x *= 10
        """
```

### 587. Erect the Fence

```python
class Solution:
    def outerTrees(self, points: List[List[int]]) -> List[List[int]]:
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (b[0] - o[0]) * (a[1] - o[1])
        
        upper, lower = [], []
        
        #1. sort points
        points.sort()
        
        #2. upper points
        for p in points:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) > 0: # cannot be equal to 0
                upper.pop()
            upper += [p]
        
        #3. lower points
        for p in points[::-1]:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) > 0:
                lower.pop()
            lower += [p]
        
        return list(map(list, set(map(tuple, lower + upper))))
    """
    Data Structure:
        1.  
    
    Algorithm:
        1.  sort points by (1) x (2) y
        
        2.  Get list of upper points
                
        3.  Get list of lower points        
        
    Input: a list P of points in the plane.

    Sort the points of P by x-coordinate (in case of a tie, sort by y-coordinate).

    Initialize U and L as empty lists.
    The lists will hold the vertices of upper and lower hulls respectively.

    for i = 1, 2, ..., n:
        while L contains at least two points and the sequence of last two points
                of L and the point P[i] does not make a counter-clockwise turn: (help < 0)
            remove the last point from L
        append P[i] to L

    for i = n, n-1, ..., 1:
        while U contains at least two points and the sequence of last two points
                of U and the point P[i] does not make a counter-clockwise turn:
            remove the last point from U
        append P[i] to U

    Remove the last point of each list (it's the same as the first point of the other list).
    Concatenate L and U to obtain the convex hull of P.
    Points in the result will be listed in counter-clockwise order.
    
    SC: O(n)
    TC: sort + stack(pop and push): O(nlogn + n)
    """
```



## Sliding Window

### 1004. Max Consecutive Ones III

```python
class Solution:
    def longestOnes(self, A: List[int], k: int) -> int:
        n = len(A)
        res = 0
        i = j = 0
        for i in range(n):
            k -= 1 - A[i]
            while k < 0:
                k += 1 - A[j]
                j += 1
            res = max(res, i - j + 1)
            
        return res
        """
        Data Structure:
            j: start of sliding window
            i: end of sliding window
            k: number of left 0
            
        Algorithm:
            1. k -= 1 - A[i]: if A[i] is 0, k -= 1
            
            2. for j in range(n):
                    (1) update k with A[j]
                    (2) while k < 0:
                            move i
                    (3) update res with i - j + 1
        
        SC: O(1)
        TC: O(N)
        """
```



## Jump Game

### Jump Game

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        leftmost = n - 1
        for i in range(0, n)[::-1]:
            if i + nums[i] >= leftmost:
                leftmost = i
        return leftmost == 0
        
        """
        Data Structure:
            leftmost: leftmose index in nums that we can start to reach n - 1
        
        Algorithm:
            1. initialize leftmost = n - 1
            
            2. iterate index from n - 2 to n:
                    if i + nums[i] >= leftmost:
                        update leftmost
        
        SC: O(1)
        TC: O(N)
        """
```



### Jump Game II

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        
        res = 1
        left = 1
        right = nums[0]
        while right < len(nums) - 1:
            nright = right
            for i in range(left, right + 1):
                nright = max(nright, i + nums[i])
            left = right + 1
            right = nright
            res += 1
        return res
                
        
        """
        Explanation:
        	
        
        Data Structure:
            1. last: furthest distance of last jump
            2. i: start index of cur jump
            
        Algorithm:
            1.  initialize last, i = nums[0], 0
            2.  while last < n - 1:
                    (1) get max jump from reachable nexr postions
                        while i < n and i <= last
                    (2) update last = new max jump
                    (3) jump += 1
        
        SC: O(1)
        TC: O(N)
        """
```



### 871. Min Number of Refueling Stops

```python
class Solution:
    from heapq import heappop, heappush
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        right_pos = startFuel
        left_idx = 0
        res = 0
        n = len(stations)
        h = []
        while right_pos < target:
            i = left_idx
            while i < n and stations[i][0] <= right_pos:
                heappush(h, -stations[i][1])
                i += 1
            if not h:
                return -1
            else:
                right_pos -= heappop(h)
                left_idx = i
                res += 1
        return res
        """
        Explanation:
            last refueling:
                s1            
            left_idx ...        right_pos
            
        
        Data Struture:
            1.  right_pos: furthest distance of last refueling(jump) 
            
            2.  left_idx: leftmost index of unvisited stations
            
            3.  h: max heap storing all visited and unused stations
            
        Algorithm:
            1.  Sort stations by postion
            
            2.  initialize right_pos = startFuel
                left_idx = 0
                res = 0
            
            3.  while last < target:
                (1) add all reachable unvisited stations to heap
                    
                    i = left_idx
                    while i < n and stations[i][0]  (pos) <= last:
                        <1> add -station[i][1] (fuel) to max heap
                        <2> i += 1
                
                (2) if not h:
                        we cannot refuel the car and move anymore:
                        return False
                
                (3) else: h is not empty --> refuel car with max fuel in heap:
                        res += 1
                        right_pos -= heappop(h)
                        left_idx = i
        
        SC: O(n) h
        TC: O(nlogn) iterate each num and push in h
        """   
```



### 1306. Jump Game III

```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        q = collections.deque()
        v = set()
        n = len(arr)
        q += [start]
        v.add(start)
        
        while q:
            i = q.popleft()
            if arr[i] == 0:
                return True
            j1, j2 = i + arr[i], i - arr[i]
            if 0 <= j1 <= n - 1 and j1 not in v:
                q += [j1]
                v.add(j1)
            if 0 <= j2 <= n - 1 and j2 not in v:
                q += [j2]
                v.add(j2)
        return False
   	
    """
   	SC: O(n) size of q
   	TC: O(n)
   	"""
```



### 1345. Jump Game IV

```python
class Solution:
    def minJumps(self, arr: List[int]) -> int:
        d = collections.defaultdict(list)
        q = collections.deque()
        v = set()
        n = len(arr)
        
        for i, a in enumerate(arr):
            d[a] += [i]
        
        q += [0]
        v.add(0)
        level = 0
        while q:
            size = len(q)
            for _ in range(size):
                i = q.popleft()
                if i == n - 1:
                    return level
                for j in [i - 1, i + 1] + d[arr[i]]:
                    if 0 <= j < n and j not in v and j != i:
                        q += [j]
                        v.add(j)
                #Notice that we reach same num for only one time, so we need to delete it
                del d[arr[i]]
            level += 1
        return -1
        """
        Data Structure:
            1.  d: {key: num in arr, val: list of index of same num}
            
            2.  q: index of same level
            
            3.  v: visited index
            
        Algorithm:  
            1.  initialize d, q and v
            
            2.  pop out pos of same level:
                (1) if pos == n - 1: return level
                (2) for pos - 1, pos + 1 and (d[arr[pos]] and not pos):
                        if valid and not visited:
                            add it to q and v
        
        SC: q + v --> O(n)
        TC: O(n)
        """
```



### 1340. Jump Game V

```python
class Solution:
    def maxJumps(self, arr: List[int], d: int) -> int:
        n = len(arr)
        @lru_cache(None)
        def dfs(i):
            res = 1
            for d1 in [1, -1]:
                for d2 in range(1, d + 1):
                    j = i + d1 * d2
                    if 0 <= j < n and arr[j] < arr[i]:
                        res = max(res, 1 + dfs(j))
                    else:
                        break
                    
            return res
        
        return max([dfs(i) for i in range(n)])
        
        
        """
        Data Structure:
            dfs(cur i): 
            max number of buildings that we can jump to (including cur i)
            with index i start point
            
        Algorithm:
            1.  dfs(cur i):
                    (1) initialize res = 1
                    (2) dir = 1 / -1
                        dis = 1 ~ x
                        
                    (3) j = i + dir * dis
                        if j in boundary and not in v:
                            <1> if j is a taller building, abandon all buildings afterwards
                            <2> if j is a shorter building, we can try to jump on it
        
            2.  for each pos:
                    update res with dfs(pos)
        
        SC: number of states: O(n)
        TC: number of states * iteration: O(n * d)
        """
```



### 1696. Jump Game VI

```python
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        dq = collections.deque()
        dq += [0]
        n = len(nums)
        for i in range(1, n):
            while dq and dq[0] < i - k:
                dq.popleft()
            
            nums[i] += nums[dq[0]]
            
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq += [i]
        return nums[-1]
            
            
        
        """
        Explanation:
            [a1, a2, ..., an-1, an] a
            before adding an,
            we pop out all elemenrts < an (useless) and index < n - k
            --> it will be a deque
            --> there is only one element or a1 > a2 > .. > an
            
            
        Data Structure:
            1.  nums[i] --> 0 ~ i - 1, covered by dp[i], max score ending with nums[i]
                             i       , still nums[i]
                             
            2.  dq: double-end queue, 
                (1) stores index of  [i - k] ~ [i - 1]
                (2) with decreasing (popleft returns max) dp
        
        Algorithm:
            1.  initialize q, res with index 0 and nums[0]
            
            2.  while q:
                (1) while index of left end of deque < i - k 
                	--> popleft biggest nums
                
                (2) update res[i] with max score in dq:
                    res[i] += res[deque[0]]
                
                (3) while corresponding num of index of right end of deque < nums[i]: 
                	--> pop useless nums
                	
                (4) update dq by adding current idx to dq
                	dq += [i]
        
        SC: dq : O(n)
        TC: push and pop each element O(n)
        """
```





### 1871. Jump Game VII

```python
from collections import deque
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        q = deque()
        n = len(s)
        
        q += [0]
        maxReach = 0
        while q:
            i = q.popleft()
            if i == n - 1:
                return True
            
            start = max(i + minJump, maxReach + 1)
            end = min(i + maxJump + 1, n)
            for j in range(start, end):
                if s[j] == '0':
                    q += [j]
            maxReach = end - 1
            
        return False
    """
    Explanation:
    	
    
    Data Structure:
        1. queue: sorted index of all reachable 0
        
        2. maxReach: replace visit; record max index of iterated pos
    
    Algorithm:
        1.  initialize q and last with first pos
         
        2.  while q:
            (1) pop out leftmsot pos
            (2) start = max(i + minJump, maxReach)
                end = min(i + maxJump + 1, n)
            (3) choose a pos in range(start, end)
                if 0 --> add it to q
            (3) update maxReach = end - 1
        return True
        
    SC: size of q : O(N) number of pos
    TC: size of q * iteration: O(N * (maxJump - minJump))
    """
```



## Spiral Matrix

### 54.Spiral Matrix

```python
class Solution:
    def spiralOrder(self, g: List[List[int]]) -> List[int]:
        m, n = len(g), len(g[0])
        r1, r2, c1, c2 = 0, m - 1, 0, n - 1
        res = []
        while r1 <= r2 and c1 <= c2:
            for i in range(c1, c2 + 1):
                res += [g[r1][i]]
            r1 += 1
            
            for i in range(r1, r2 + 1):
                res += [g[i][c2]]
            c2 -= 1
            
            if r1 > r2 or c1 > c2:
                break
            
            for i in range(c1, c2 + 1)[::-1]:
                res += [g[r2][i]]
            r2 -= 1
            
            for i in range(r1, r2 + 1)[::-1]: 
                res += [g[i][c1]]
            c1 += 1
        return res
    """
    Data Structure:
        r1, r2, c1, c2
        current loop of nums that we are going to iterate
        
    Algorithm:
        1.  while r1 <= r2 and c1 <= c2:
            (1) iterate r1 from c1 to c2: r1 += 1
            
            (2) iterate c2 from r1 to r2: c2 -= 1
            
        2.  if r1 <= r2 and c1 <= c2:
            (1) iterate r2 from c2 to c1
            
            (2) iterate c1 from r2 to r1
            
    SC: O(m * n)
    TC: O(m * n)
    """
```

## Intervals

### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        intervals = sorted(intervals) + [[10 ** 4  + 1, 10 ** 4  + 1]]
        ns, ne = intervals[0] 
        
        for s, e in intervals[1:]:
            if ns <= s <= ne:
                ne = max(ne, e)
            else:  
                res += [[ns, ne]]
                ns, ne = s, e
        return res
    """
    Explanation:
    	res[<s1, e1> <s2, e2> ...] <ns, ne>
    	
    Data Structure:
        1.  intervals: sorted by start 
        
        2.  new interval with new start and new end  
    
    Algorithm:
        1.  sort intervals and add [0, 0] to end
        
        2.  initialize new start and new end with first interval
        
        3.  for each interval in [1:]:
                if new_s <= s <= new_e:
                        update new_e with max(e, new_e)
                else:
                    add new interval to res
                    initialze a new interval with cur interval
   	SC: O(1)
   	TC sort + iteration O(nlogn + n)
    """
```



### 57.Insert Interval

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        i = 0
        n = len(intervals)
        while i < n and intervals[i][1] < newInterval[0]:
            res += [intervals[i]]
            i += 1
        
        while i < n and newInterval[1] >= intervals[i][0]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
            
        res += [newInterval]
        res += intervals[i:]
        return res
        
        """
        Explanation:
            There are 3 situations of relationship between old interval and new interval
            (1) prev        <old start --- old end>,  <new start --- new end>
                old end < new start
                
            (2) overlap     
                <1> old end >= new start
                and
                <2> new end >= old start
                
                Actually there are also 3 situations:
                <1> old     |_______|
                    new         |_______|
                
                <2> old     |_________|
                    new         |___|
                    
                <3> old     |________|
                    new  |_______________|
            
            (3) after       <new start --- new end>, <old start --- old end>
                new end < old start
        
        Data Structure:
            (1) intervals sorted by start
            (2) new interval: new start and new end
            
        Algorithm:
            1.  add all previous intervals to res
                    while old end < new start:
                        add it to res
            
            2.  (1) for all overlap intervals:
                        while new end >= old start
                            update new interval:
                                new start = min(old start, new start)
                                new end = max(old end, new end)
                
                (2) add new interval to res
                
            3.  for all after intervals:
                    add them to res
        
        SC: O(N)
        TC: O(N)
        """
```



### 253. Meeting Rooms II

```python
class Solution:
    from heapq import heappush, heappop
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        h = []
        for s, e in intervals:
            heappush(h, (s, 1))
            heappush(h, (e, -1))
            
        cur = 0
        res = 0
        while h:
            t1, t2 = heappop(h)
            if t2 == 1:
                cur += 1
                res = max(res, cur)
            else:
                cur -= 1
        return res
        """
        Data Structure:
            heap: [time, start / end]
        
        
        Algorithm:
            1. add each element [time, 1 or -1] to h
            
            2. for each element:
                    if start:
                        cur += 1
                        update res
                    else if end:
                        cur -= 1
        
        SC: size of h O(n)
        TC: push in h and pop out h--> O(nlogn + n)
        """
```

### 436. Find Right Interval

```python
class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        res = []
        starts = []
        d = {}
        for i, (s, _) in enumerate(intervals):
            starts += [s]
            d[s] = i
        
        starts.sort()
        n = len(intervals)
        for _, e in intervals:
            l, r = 0, n
            while l < r:
                m = l + (r - l) // 2
                if starts[m] >= e:
                    r = m
                else:
                    l = m + 1
            if l == n:
                res += [-1]
            else:
                res += [d[starts[l]]]
        return res
        
        """
        Explanation:
        	e 	s1, s2, ... ,sn
        		i1, i2,. .., in  in d
        		
        	find idx of smallest s1 >= e --> if cannot. -1
        	
        	
        Data Structure:
            1.  sorted list of starts
            
            2.  d[key: start, val: index in original array]
        
        Algorithm:
            1.  initialize starts and d
            
            2.  for each end of interval in intervals:
                    (1) binary search first start >= end in list of starts
                    (2) add index of first start to res
        
        SC: O(N)
        TC: sort + iteration * Binary Search O(nlogn + nlogn)
        """
```



## Populating Next Right Pointers in Each Node 

### Populating Next Right Pointers in Each Node

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return
        if root.left:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
        
        self.connect(root.left)
        self.connect(root.right)
        
        return root
        """
        Explanation:
        		cur					cur.next
        	left	right		left
        	
        	as we can see if we want to get left -> right -> left
        	cur node must be 
        	
        	since it is a perfect tree, if cur.next and cur.left --> cur.right must exist
        	
        Data Structure:
            dfs(node):
            	(1) node.next is known and we are going to connect node.left and node.right
                (2)	return root whose left.next and right.next has been assigned
        
        Algorithm:
            1.  connect root.left with root.right
            
            2.  if root.left has been connected 
                connect root.right with root.next.left
            
            3.  connect left subtree and right subtree
            
        Follow up 117
        SC: O(n) depth of recursion tree
        TC :O(n) number of nodes
        """
```

### Populating Next Right Pointers in Each Node II

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        node = root
        while node:
            dummy = tail = Node(0)
            cur = node
            while cur:
                if cur.left:
                    tail.next = cur.left
                    tail = tail.next
                if cur.right:
                    tail.next = cur.right
                    tail = tail.next
                cur = cur.next
            node = dummy.next
        return root
        """
        Explanation:
        				node			node.next
        	dummy	left	right	left		right
        	
        	Each time
        	we start from head in a linked list 
        	move on to next
        	
        	Each time
        	add left / right of cur node to dummy if not empty
        	move on to next
        	
        
        Data Structure:
            1.  node: head of LinkedList in current row  
            
            1.  dummy: head of LinkedList with all connected nodes in next row
        
        Algorithm:
            1. create a new dummy
            
            2. while node is not None
                    (1) add node.left to dummy
                    
                    (2) add node,right to dummy
                    
                    (3) node = node.next
                
            3. node = dummy.next
            
        SC: O(n) depth of recursion tree
        TC :O(n) number of nodes
        """
```



## PreSum

### 303. Range Sum Query

```python
class NumArray:
    def __init__(self, nums: List[int]):
        n = len(nums)
        self.preSum = [0] * (n + 1)
        for i in range(1, n + 1):
            self.preSum[i] = self.preSum[i - 1] + nums[i - 1]

    def sumRange(self, left: int, right: int) -> int:
        return self.preSum[right + 1] - self.preSum[left]


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)

    """
    Data Structure:
        presum[i] : num[0] ~ num[i - 1]
        presum[0] --> 0
    
    SC: init O(n) sumRange O(1)
    TC: init O(n) sumRange O(1)
    """
```

### 304. Range Sum Query 2D - Immutable

```python
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        self.dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self.dp[i][j] = matrix[i - 1][j - 1] + self.dp[i - 1][j] + self.dp[i][j - 1] - self.dp[i - 1][j - 1]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.dp[row2 + 1][col2 + 1] - self.dp[row2 + 1][col1] - self.dp[row1][col2 + 1] + self.dp[row1][col1]


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
    
    """
    303 preSum
    Data Structure:
        dp[i][j] --> sum from 0, 0 to i - 1, j - 1
    
    SC: init O(n * n) sumRange O(1)
    TC: init O(n * n) sumRange O(1)
    """
```

### 523. Continuous Subarray Sum

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        d = {0 : -1}
        n = len(nums)
        pre = 0
        for i, n in enumerate(nums):
            pre += n
            r = pre % k
            if r in d:
                l = i - (d[r] + 1) + 1
                if l >= 2:
                    return True
            else:
                d[r] = i
        return False
    
    """
    Explanation:
    			0 ~ i ~ j
        		sum of 0 ~ i % target --> r
        		sum of 0 ~ j % target --> r
        		
        		--> sum of i + 1 ~ j has % target == 0
        		
    Data Structure:
        d: {key: remainder of preSum, val: first  last pos of end num}
        
    Algorithm:
        1.  put {0 : -1} into d (Base Case)
        
        2.  calculate cur preSum and its remainder
        
        3.  if remainder exists in d:
                len = i - (d[i] + 1) + 1
                if len >= 2
                    return True
            else:
                d[remainder] = cur pos
                
    SC: d O(target)
    TC: iteration O(N) 
    """
                
```

### 1074. Number of Submatrices That Sum to Target

```python
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:               
        m, n = len(matrix), len(matrix[0])
        preSum = [[0] * (n + 1) for _ in range(m)]
        for i in range(m):
            for j in range(1, n + 1):
                preSum[i][j] = matrix[i][j - 1] + preSum[i][j - 1]
        
        res = 0
        for c1 in range(1, n + 1):
            for c2 in range(c1, n + 1):
                d = defaultdict(int)
                d[0] = 1
                total = 0
                for r in range(m):
                    num = preSum[r][c2] - preSum[r][c1 - 1]
                    total += num
                    res += d[total - target]
                    d[total] += 1
        return res
        """
        Explanation:
        	c1 ... c1			c1
        	c2 ... c2 	--> 	c2
        	c3 ... c3			c3
        	
        	By merging cols together, count submatrices --> count preSum
        	
        	
        		
        Data Structure:
            1.  preSum[i][j]: preSum array for each row, for row i, preSum from 0 ~ j
                --> merge all elements between two cols into one element
                
            2.  d: 
                how many time new element shows up 
                {key: preSum, val: times}
                
            3.  cur:
                sum of previous new element
                
            4.  new element: sum of all num between col1 and col2 in a row
        
        Algorithm:
            1.  initialize preSum of each row d[0] = 1
                for each row i
                    for each col j:
                        preSum[i][j] = A[i][j] + preSum[i][j - 1]
            
            2.  (1) fix two col i, j (merge all elements between them into one element)
                (2) initialize d {key: 0, val: 1}
                (3) for each row k, new element should be preSum[k][j] - preSum[k][i - 1]
                        <1> update cur
                        <2> update res with d[cur - target]
                        <3> update d[cur]
        """
```



### 1314. Matrix Block Sum

```python
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        preSum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                preSum[i][j] = mat[i - 1][j - 1] + preSum[i][j - 1] + preSum[i - 1][j] - preSum[i - 1][j - 1]
        
        res = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                r1, c1, r2, c2 = max(0, i - k), max(0, j - k), min(m - 1, i + k), min(n - 1, j + k)
                res[i][j] = preSum[r2 + 1][c2 + 1] - preSum[r2 + 1][c1] - preSum[r1][c2 + 1] + preSum[r1][c1] 
        return res
        """
        Explanation:
        	for matrix whose center is (i, j) 
        	bottom-right --> (i + k, j + k)
        	top-left --> (i - k, j - k)
        
        Data Structure:
            preSum[i][j]: sum of all nums from 0, 0 to i - 1, j - 1
        
        Algorithm:
            1.  initialize preSum[i][j]
            
            2.  for each pos in preSum: (1 ~ m, 1 ~ n)
                    get 4 coordinates and compute sum:
                        r1, c1: min should be 1 
                        r2, c2: max should be m or n
        
        """
```



## Dijkstra

### 1514. Path with max Probability

```python
class Solution:
    from collections import defaultdict
    from heapq import heappush, heappop
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        h = []
        hv = [0] * n
        g = defaultdict(list)
        
        for i, (a, b) in enumerate(edges):
            prob = succProb[i]
            g[a] += [(b, prob)]
            g[b] += [(a, prob)]
        
        heappush(h, (-1, start))
        hv[start] = 1
        
        while h:
            p, i = heappop(h)
            if i == end:
                return -p
            for j, np in g[i]:
                if -np * p > hv[j]:
                    heappush(h, (np * p, j))
                    hv[j] = -np * p
        return 0
        """
        For Dijkstra algorithm, each node has attribute cost
        
        	If we only set nodes as visited when poping it out from heap
            we may push too many nodes into heap before get that position
            
            In this way, visited need to record cost of nodes that we push into heap 
            and we only push those nodes whose cost < record into heap
            
            
        Data Structure:
            1.  h: max heap, (prob, end): prob to reach end
            
            2.  graph[i]: lsit of (j, p), prob from i to reach j
            
            3.  v[i]: max prob to reach i (update if added to heap)
        
        Algorithm:
            1.  initialize h with [1, start]
                initialize g with [next node, prob]
                initialize v[start] = 1
            
            2.  while h:
                (1) pop out i with most prob
                (2) if i == end:
                        return prob
                (3) for next node j of i:
                        get new prob
                        if new prob > max prob to reach j:
                            add it to h
                            v[j] = new prob
        
        SC: for edge a -> b, though there are many edges connecting a and previous nodes, we only need to consider the smallest one
            size of h --> number of edges
            size of d --> number of vertex
            O(V + E)
        
        TC: h: for each element in h, push all its edges into h:
            d: for each edge --> O(E)
            O(ElogE)   In the worst case, O(E) = O(V^2), so O(logE) = O(log(V^2)) = 2 O(logV) = O(logV). 
            O(ElogE) --> O(ElogV)
            
            O(ElogV)
            
        """
```

### 1928. Minimum Cost to Reach Destination in Time

```python
class Solution:
    from heapq import heappush, heappop
    def minCost(self, maxTime: int, edges: List[List[int]], costs: List[int]) -> int:
        n = len(costs)
        g = defaultdict(list)
        hv_t = [inf] * n
        hv_c = [inf] * n
        
        for a, b, t in edges:
            g[a] += [(b, t)]
            g[b] += [(a, t)]
        
        h = []
        heappush(h, (costs[0], 0, 0))
        
        while h:
            c, i, t = heappop(h)
            if i == n - 1:
                return c
            for j, dt in g[i]:
                nc, nt = c + costs[j], t + dt
                if nt <= maxTime and (hv_t[j] > nt or hv_c[j] > nc):
                    heappush(h, (nc, j, nt))
                    hv_t[j] = min(hv_t[j], nt)
                    hv_c[j] = min(hv_c[j], nc)
        return -1
    
    """
    Explanation:
        For this Dijkstra, each node has two attribtue cost(main) and time
        
        when we have new cost and new time of a node:
        
        we push it into heap if and only if:
        
        not visited or smaller cost or smaller time --> set unvisited as inf
    
    Data Structure:
        1.  heap[cur cost, cur end, cur time]
        2.  hv_t{key: node, val: min time to reach node}
            hv_c{key: node, val: min cost to reach node}
            
        3.  g[i]: {key: cur node, val: list of (next point, time to travel)}
        
    Algorithm: 
        1.  initialize g
        
        2.  add start point and its passing fee to h
        
        3.  while h:
                (1) pop out node with min cost
                
                (2) if end point is target: return cost
                
                (3) for each next node j of cur node i:
                        if new time < max time and (new time is smaller or new cost is smaller)
                            <1> update v[j]
                            <2> add j, time, cost to heap
                            
    SC: O(E + V)
    TC: O(ElogV)
    """
```

TC: O (VlogV + E) 

SC: O(V(h) + E(g and hv))





