#  Amazon Leetcode



## HashTable(dictionary & set)



### 两个数组元素是否相同

如果没有重复元素：

先判断长度； 如果长度相同，把一个数组转化为hashset, 对另外一个数组进行遍历在hashset里查找，这样o(n)时间内可以完成。

如果有重复元素：用 HashMap 记录 (num, freq) 数组 A 的freq, 然后遍历另一个数组，相应 num 的 freq 减一，最后遍历 HashMap, 看所有的数的 freq 是否都是 0



### 1.Two Sum

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, num in enumerate(nums):
            if target - num in d:
                return [i, d[target - num]]
            else:
                d[num] = i
        return []
        
        """
        Data Structure:
            d:  [key: num, val: index]
                store index of num of all previous nums
            
        Algorithm:
            iterate each num:    
                (1)if target - num exists in d --> we have a complete pair
                        return [i, d[target - i]]
                (2)else not exist in d: --> put it into d
        
        """
```



### 14.Longest Common Prefix

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        for chars in zip(*strs):
            s = set(chars)
            if len(s) > 1:
                break
            else:
                res += chars[0]
        return res
        
        """
        Data Strucutre:
            1.zip(*strs): str --> list
             i      0  1  2  3  4  5
             
             0      f  l  o  w  e  r
             1	    f  l  o  w
             2	    f  l  i  g  h  t
            
        Algorithm:
            1. convert all strings into list and zip them together
            
            2. for each chars in one postion of zip:
                    (1) if num of distinct chars > 1:
                        this char cannot be in prefix
                        break
                    
                    (2)else: if num of different chars is 1:
                        add it to prefix
            
            3. return prefix
        """
```





### 828.Count Unique Characters of All Substrings of a Given String

```python
class Solution:
    def uniqueLetterString(self, s: str) -> int:
        d = {}
        for c in string.ascii_uppercase:
            d[c] = [-1, -1]
        
        res = 0
        for i, c in enumerate(s):
            i1, i2 = d[c]
            i3 = i
            res += (i3 - i2) * (i2 - i1)
            d[c] = [i2, i3]
        # after iteration
        for c in string.ascii_uppercase:
            i1, i2 = d[c]
            i3 = len(s)
            res += (i3 - i2) * (i2 - i1)
            
        return res
        """
        	count unique A2: A1 B C A2 C B A1
        
            Explanation:
            A1 B C A2 C B A3
            
            how many substring can unique A2 show up
            
            Imagine to insert '(' and ')'
            
            --> times = number of '(' * number of ')'
            
            image now move to char C and we have last two postions of this Char
            d[C] = {i1, i2}, current index is i3
            
            In this way, we can set char C on pos i2 as unique char
            number of subarray where char C is unique == (i2 - i1) *(i3 - i2)
            
            After Iteration,
            we have d[C] = {p1, p2}
            char C on pos p2 has not been set unique yet
            so we now need to deal with it by setting i1 = p1, i2 = p2, i3 = len(s)
            res += (i2 - i1) * (i3 - i2)
            
        Data Structure:
            1.  d[c] = {last but one pos, last pos}
            	(1) initialize each d[c] = [-1, -1] (because first idx is 0)
            	(2) update, after updating res, set d[c] = [old last pos, cur pos]
                
        Algorithm:
            1.  initialize d[each letter] = [-1, -1]
            
            2.  for each c in s:
                    (1) (i1, i2), i3 = d[c], current index i 
                    (2) res += (i2 - i1) * (i3 - i2)
                    (3) update last two positions:
                        d[c] = {i2, i3}
            
            3.  for each letter in d:
                    i1, i2 = d[letter]
                    i3 = n 
                    res += (i2 - i1) * (i3 - i2)
        """
```





### 1152.Analyze User Website Visit Pattern



```python
from collections import defaultdict
from itertools import combinations
class Solution:
    def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
        A = sorted(zip(timestamp, username, website), key = lambda x : x[0])
        d1 = defaultdict(int)
        d2 = defaultdict(list)
        for t, u, w in A:
            d2[u] += [w]
        
        for ws in d2.values():
            ps = combinations(ws, 3)
            for p in set(ps):
                d1[p] += 1
        
        res = sorted(d1.keys(), key = lambda x : (-d1[x], x))
        return res[0]
        """
        Explanation:
            d[user1]: [w1, w2, w3, ...]
            d[user2]: [w2, w3, w4, ...]
            
            --> combination of websites = pattern
            
            d[user1]: [pattern1, pattern2, pattern3, ...]
            d[user2]: [pattern1, ...]
            
            --> count # of pattern
            
            count[pattern] = #
 
        
        Data Structure:
            1. d1: key: pattern(tuple), val: number of users (int)
            2. d2: key: user(string) val: list of websites(list) 
            
        Algorithm:
            1.  get zip(time, user, website) sorted by time
            
            2.  for each t, u, w in zip(,,,):
                    d2[u] += [w]
                    
            3.  for each list of ws (values() of d2):
                    (1)patterns = set(combinations(list of ws, 3))
                    (2)for each pattern p
                            d1[p] += 1
            
            4.  sort pattern by (1) freq (2) lexicographically(if freq is equal)
                res = sorted(d1.keys(), key = lambda x : (-d1[x], x))
                return res[0]
        """
```





### 1481.Least Number of Unique Integers after K Removals



```python
class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:  
        d = collections.Counter(arr)
        res = len(d)
        
        for freq in sorted(d.values()):
            if k > freq:
                res -= 1
                k -= freq
            elif k == freq:
                res -= 1
                return res
            else:
                return res
        return res
        """
        Data Structure:
            1.  d: {key: num, val: freq} sorted by freq
            	(1) + ：initialize each int with its freq
            	(2) - : each time remove the pair with min freq while k > 0
        
        Algorithm:
            1.  count freq of each num
            2.  extract values (freq) and sort it
            3.  total number of unique number
                for v in sorted values:
                    if freq < k: current number can be completely deleted and we still need to delete other number 
                        k -= freq
                        total -= 1
                    elif freq == k: current number can be completely deleted and stop
                        total -= 1
                        return total
                    else: we cannot delete all of current numbers so stop
                        return total     
        """
```







## Union Find

### 1135.Connecting Cities With Minimum Cost

```python
from heapq import heappop, heappush
class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        p = [0] * n
        for i in range(n):
            p[i] = i
        
        def find(node):
            if p[node] == node:
                return node
            p[node] = find(p[node])
            return p[node]
        
        h = []
        for a, b, c in connections:
            heappush(h, (c, a - 1, b - 1))
        
        res = 0
        k = n - 1
        while h and k > 0:
            c, a, b = heappop(h)
            p1, p2 = find(a), find(b)
            if p1 == p2:
                continue
            else:
                res += c
                p[p1] = p2
                k -= 1

        return res if k == 0 else -1
        """
        Explanation:
            choose n - 1 edges starting from min cost
            
        Data Struture:
            1.  h : min heap [cost, a, b]
            	(1) +: initialize 
            	(2) -: each time pop out the one with min cost
            
            2.  p: parent of each node
            	(1) +: initialize each node's parent is itself
            	(2) -: 
            	
        Algorithm:
            1.  initialize h with each edges
            
            2.  pop out edges with min cost
                    p1, p2 = find(a), find(b)
                    (1)if p1 == p2:
                            a and b have been connected, they are in the same tree
                            we do not use this edge
                    (2)if p1 != p2:
                            connect a and b
                            p[p1] = p2
            
            3.  how to judge whether this is only one tree?
                if p[node] == node: node is root
                check how many roots are in parent
        """
```







## Binary Search



### 数组中和最接近0的两个数

binary search

TC: O(nlogn)

```python
def findclosest(arr):
    arr.sort()
    n = len(arr)
    res = [10000, 10000]
    for i in range(n):
        num = arr[i]
        l, r = 0, n - 1
        while l + 1 < r:
            m = l + (r - l) // 2
            total = num + arr[m]
            if total == 0 and m != i:
                return [arr[i], arr[m]]
            elif total < 0:
                l = m
            else:
                r = m
            #decide which is closer to 0 ? arr[i] + arr[l] / arr[i] + arr[l]
        i1 = i; i2 = -1
        if l == i or r == i:
            i2 = l if l != i else r
        else:
            i2 = l if abs(arr[i] + arr[l]) < abs(arr[i] + arr[r]) else r
        if abs(sum(res)) > abs(arr[i1] + arr[i2]):
            res = [arr[i1], arr[i2]]
    return res
    
arr = [-50, -40, 100, 200]
print(findclosest(arr))
```





### [33. Search in Rotated Sorted Array](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)



```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        n = len(nums)
        l, r = 0, n - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m
            # check if left part
            elif nums[l] <= nums[m]:
                # check if target in [left, mid] -> continuously increasing part
                if nums[l] <= target <= nums[m]:
                    r = m - 1
				# target in [mid, min, right] -> incontinously increasing part
                else:
                    l = m + 1
			# check if right part
            else:
                if nums[m] <= target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return -1 if nums[l] != target else l
```





### 81.Search in Rotated Sorted Array II  

本题含有重复元素

这个题目就比[33. Search in Rotated Sorted Array](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)的不含重复元素的题目多了一个去除某些重复元素的情况，当 nums[mid] == nums[right] 时，让 right -= 1，并退出本次循环（continue），其余部分完全相同。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        n = len(nums)
        l, r = 0, n - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return True
            # try to keep nums, which same as nums[m], on one side
            elif nums[m] == nums[r]:
                while l <= r and nums[r] == nums[m]:
                    r = r - 1
                continue
            elif nums[l] <= nums[m]:
                if nums[l] <= target <= nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                if nums[m] <= target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return False
```



注 Attention：nums[left] > nums[mid] 说明后半部分有序，反之前半部分有序。恰好相等的时候，你就不知道哪半部分有序了，所以需要 left ++，因为 nums[mid] == nums[left]，所以也不用害怕，left ++把需要找的数加没了。



### 1268.Search Suggestions System

```python
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        prefix = ''
        n = len(products)
        products.sort()
        res = []
        
        for c in searchWord:
            prefix += c
            l, r = 0, n - 1
            while l < r:
                m = l + (r - l) // 2
                if products[m] >= prefix:
                    r = m
                else:
                    l = m + 1
                    
            suggestWords = []
            for i in range(3):
                if l + i <= n - 1 and products[l + i].startswith(prefix):
                    suggestWords += [products[l + i]]
                else:
                    break
            res += [suggestWords]
        return res
                        
        '''
        Data Structure:
            1.  suggestedProduct(prefix):
                +: choose 3 least product lexicographically starts with prefix
            2.	prefix:
            	+: each time, append one char from searchWord to prefix, find least bigger word in products through binary search
        
        Algorithm:
            1.  sort products
            2.  for each prefix of searchWord:
                (1) find least product >= prefix
                (2) iterate 3 products, if product.startWith(prefix), add it to suggestWords
        '''
```



## Stack

### 907.Sum of Subarray Minimums

```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        mod = 10 ** 9 + 7
        res = 0
        stack = []
        arr = [0] + arr + [0]
        
        for i in range(len(arr)):
            while stack and arr[stack[-1]] > arr[i]:
                i2 = stack.pop()
                i1 = stack[-1]
                i3 = i
                res += (i2 - i1) * (i3 - i2) * arr[i2]
            else:
                stack += [i]
        return res % mod
        
        """
        LC 84, 85, 907 
        Explanation:
            1.
            j     <      i    >       k
            i1          i2            i3
            j is first num < i on the left: j is left boudnary of i
            k is first num < i on the right k is right boundary of i
            all left boudnaries and min num are stored in stack
            k is idx of cur num
            
            num of subarray whose minimum is i --> (i - j) * (k - i)
            
            2.stack
            (1) pop: arr[stack[-1]] > arr[i] 
            (2) push: after popping out all idx in stack
            
            stack[num1, num2, num3] num 4
            Assume we can pop out num2 and num3
            <1> num2, num3 > num4 => num4 is first num < num2 and num3
            <2> num1 < num4 => num1 is first num < num4
            <3> num2 and num3 can be pushed w/o popping out num1 => 
            	num1 is first num < num2
            	num2 is first num < num3
            
            In this way, we can update res when we popping out num2 and num3 by regarding them as i2
            
            3. modulo: 取余数
		Data Structure:
			1.	stack: left boundary + idx of min num
				(1) +: for each iteration, add one num to end of stack
				(2) -: pop out all min nums in stack whose right boundary is cur idx 
				
        Algorirhtm:
            1.  arr = [0] + arr + [0]
            
            2.  iterate each num from left to right:
                    (1) while (stack[-2] <) stack[-1] > num:
                        i = stack.pop()
                        j = stack[-1]
                        k = cur index
                        update res with i, j, k and num
                        
                    (2) when we cannot pop stack anymore
                        (stack[-2] <) stack[-1] < num
                        stack[-1] will be first num smaller than cur num:
                        
                        add cur num to stack
                        
        """
```



#### 239.Sliding Window Maximum

```python
from collections import deque
class Solution:
	def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        res = []
        for i, num in enumerate(nums):
            while dq and nums[dq[-1]] < num:
                dq.pop()
            while dq and i - dq[0] + 1 > k:
                dq.popleft()
            dq += [i]
            if i >= k - 1:
                res += [nums[dq[0]]]
        return res
        '''
        Explanation:
        	1.
        	dq: [left boundary, max num] cur num(right boundary)
			left boundary: first bigger num on the left 
			right boundary: first bigger num on the right
				
			if we can fix boundary of 1 num
			it can no longer become max num in a window
			
			dq is to find boundary of max, so it decrease from max to min
			
			2.
			if one num is too far away from cur num, it cannot becomes max num either
			
        Data Structure:
            1.  dq:
                (1) pop: while rightmost num in dq < cur num, pop out rightmost num
                (2) popleft: while length of sliding window > k after adding cur num, pop out leftmost num
                (3) push: after popping out rightmost and leftmost num, add idx of cur num to dq
                
            2.  res:
                +: if idx of rightmost num in dq >= k - 1, add max num in dq, namely dq[0], to res
        '''
    """
```



### 735.Asteroid Collision

https://leetcode.com/problems/asteroid-collision/

------



```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        res = []
        for a in asteroids:
            if a > 0:
                stack += [a]
            else:   
                killed = False
                while stack:
                    if stack[-1] > a:
                        killed = True
                        break
                    elif stack[-1] == a:
                        killed = True
                        stack.pop()
                        break
                    else:
                        stack.pop()
                if not killed:
                    res += [a]
        res += stack
        return res
        
        '''
        Explanation:
            stack.pop => stack[-1] > 0 and abs(asteroids[i]) >= satck[-1]
            stack.push => asteroids[i] > 0
            
            res add new asteroids:
            (1) moving left and kill all asteroids moving right
            (2) moving right and kill all asteroids moving left
            
        Data Structure:
            1. stack: all remaining asteroids moving right
            (1) +: 	if cur asteroid moves right, add it to stack directly
            		if cur asteroid moves left and all asteroids moving right in stack and be killed by it, add it to stack
            (2) -:	if cur asteroid moves left, pop out asteroids moving right as much as possible 
            
            2.: true if current asteroid exists
        
        Algorithm:
            1.  iterate each asteriod:
                (1)  if moving right:
                        add it to stack
                (2)  if moving left:
						<1> while stack is not empty:
                            if top of stack > cur: cur will die and break loop
                            elif top of stack == cur: cur will die, top of stack will die too, break
                            else top of stack < cur: top of stack will die and continue
                     	<2> if asteroid moving left is still alive, add it to res
                
            2.  if asteroids moving right still alive, add them to res
        '''
```





## BFS



### 102.Binary Tree Level Order Traversal

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = deque()
        queue += [root]
        res = [] 
        while queue:
            size = len(queue)
            temp = []
            for _ in range(size):
                node = queue.popleft()
                temp += [node.val]
                if node.left:
                    queue += [node.left]
                if node.right:
                    queue += [node.right]
            res += [temp]
        return res
        '''
        Data Structure:
            1. queue
                (1) pop: remove node in cur layer
                (2) push: add nodes in next layer
            
            2. temp
                +: iterate node in one layer along with pop
            
            3. res
                +: when whole one layer of nodes have been added
        '''
```





### 127.Word Ladder

```python
from collections import deque
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordList = set(wordList)
        if endWord not in wordList:
            return 0
        q = deque()
        q += [beginWord]
        level = 1
        
        while q: # q 不为空
            size = len(q)
            for _ in range(size):
                w = q.popleft()
                if w == endWord:
                    return level
                for i in range(len(w)):
                    for j in range(26): # iterate all letters
                        c = chr(ord('a') + j) 
                        nw = w[:i] + c + w[i + 1:] # w[i] is replaced by c
                        if nw in wordList:
                            q += [nw]
                            wordList.remove(nw)
            level += 1
        return 0
        """
        Data Structure:
            1.	q, queue
            	(1) pop: iterate words in one level
            	(2) push: iterate words in one level and push words of next level
            	
            2. 	wordList:
            	delete: push words of next level, if word in wordList, delete it and push it into queue
            
        Algorithnm: BFS
            1. add beginWord to q:
            2. while queue is not empty:
                    check each word in the same level / q
                    
                    (1) if word is endWord:
                            return level
                            
                    (2) not:
                        for each char in word:
                            char --> 'a' ~ 'z'
                            if new word in wordList
                                add newWord to queue
                                delete newWord from wordList
                    (3) after checking all words:
                            level += 1
        """
```







### 1730.Shortest Path to Get Food



```python
from collections import deque
class Solution:
    def getFood(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        queue = deque()
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '*':
                    queue += [[i, j]]
                    break
        step = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                i, j = queue.popleft()
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] in ['#', 'O']:
                        if grid[ni][nj] == '#':
                            return step + 1
                        grid[ni][nj] = '|'
                        queue += [[ni, nj]]
            step += 1
        return -1
    '''
    Data Structure:
        q: lastly added nodes of same level
    
    '''
```





### 994.Rotting Oranges



```python
from collections import deque
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid: # if grid is None
            return -1
        m, n = len(grid), len(grid[0])
        fresh = 0
        q = deque()
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1
                elif grid[i][j] == 2:
                    q += [(i, j)]
        
        time = 0
        while q and fresh > 0:
            size = len(q)
            for _ in range(size):
                i, j = q.popleft()
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= m or nj < 0 or nj >= n or grid[ni][nj] in [0, 2]:
                        continue
                    fresh -= 1
                    grid[ni][nj] = 2
                    q += [(ni, nj)]
            time += 1
        return time if fresh == 0 else -1
    
# Time complexity: O(rows * cols) -> each cell is visited at least once
# Space complexity: O(rows * cols) -> in the worst case if all the oranges are rotten they will be added to the queue
```







## DFS

### 79.Word Search   DFS

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
            	(1) base case: if len(w) == 0, return True
                (2) recursion: iterate neighbor cell to check if word[1:] match
                (3) return: true if one direction of recursion can match 
                
            2. board:
                (1) set to # when recursion to avoid dfs back
                (2) set back after recursion
        '''
```





### 212.Word Search II

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
                (1) recursion: iterate neighbor chars 
            
            3.  grid[i][j]
                (1) set to '#': 
                (2) set back:
            
            4.  res:
                +: dfs(i, j, node), node.word == True
                
        Algorithm:
            1.  initialize TrieNode
                (1) build a new class TrieNode
                (2) add all words to Trie
                now we have root of Trie (t)
            
            2.  (1) for each pos:
                        if cur char(grid[i][j]) on current pos in children of t:
                            dfs(i, j, t.children[cur char])
                
                (2) de-duplicate res
                        
                        
            3.  dfs(i, j, node):
                    (1) if node.word is not None:
                            add its corresponging word to result
                        
                        #Notice that we cannot return / end here
                        #eat --> eateat 
                        #so we need to conitinue to search
                    
                    (2) <1> convert it into '#' as visited
                        <2> for new pos ni, nj:
                            if not out of boundary and grid[ni][nj] in children of node:
                                dfs(ni, nj, node.children[grid[ni][nj](next char)])
                        <3> convert it back
        """   
```





### 200.Number of Islands

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
                	after converting current cell into water
                	iterate neighbor cells to convert them into water
            
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





### 472.Concatenated Words



```python
from functools import lru_cache
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        wordSet = set(words)
        @lru_cache(None)
        def dfs(w):
            if len(w) < 2:
                return False
            res = False
            for i in range(1, len(w)):
                w1, w2 = w[:i], w[i:]
                res |= (w1 in wordSet or dfs(w1)) and (w2 in wordSet or dfs(w2))
            return res
        
        res = []
        for w in words:
            if dfs(w):
                res += [w]
        return res
            
        '''
        Data Structure:
            1. dfs(w)
            	(1) if len(w) < 2, return True 
                (2) recursion: iterate w to divide it into two <1> word in wordSet <2> a dividible word
                (3) return: True if w is dividible
            
            2. wordSet:
                -: try to divide one word in words
        '''
```





## Tree



### 103.Binary Tree Zigzag Level Order Traversal

Tree + BFS(flag)

------



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





### 236.Lowest Common Ancestor of a Binary Tree



```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node, p, q):
            if node is None:
                return None
            if node.val == p.val or node.val == q.val:
                return node
            l, r = dfs(node.left, p, q), dfs(node.right, p, q)
            if l and r:
                return node
            elif not l:
                return r
            elif not r:
                return l
        return dfs(root, p, q)
        '''
        Data Structure:
            1.  dfs(node, p, q)
            	(1) base case: if node is None, return None
                (2) recursion: if node is not p & q, dfs(node.left), dfs(node.right)
                (3) return: 
                    <1> if node is p or q, return p / q
                    <2> if node.left and node.right both can find p / q, return node
                    <3> if only node.left / node.right can find one p / q, return p / q
		
		Algorithm:
			1.	check if node is None
			2.	check if node is p / q
			3. 	dfs(node.left) and dfs(node.right)
				check if node is parent / single p, q / Nothing
        '''
```



### 101.Symmetric Tree



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True
        return self.isMirror(root.left, root.right)
        
    def isMirror(self, root1, root2):
        if root1 is None and root2 is None:
            return True
        elif root1 is None or root2 is None:
            return False
        return root1.val == root2.val and self.isMirror(root1.left, root2.right) and self.isMirror(root1.right, root2.left)
    
    '''
    Data Structure:
        1.  isMirror(root1, root2):
        	(1) base case:
        		<1> if root1 and root2 are both None, return True
                <2> if only one of root1 and root2 is None, return False
            (2) recursion: 
            	if root1 == root2
            	<1> dfs(root1.left. root2.right)
            	<2> dfs(root1,right, root2.left)
            (3) return: True if one recursion succeeds
	
	Algorithm;
		1.	check none
		2.	check if (1) can recursion (2) recursion succeeds
    '''
```





### 863.All Nodes Distance K in Binary Tree

https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/

------

BFS + DFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        d = defaultdict(list)
        def dfs(p, c):
            if c is None:
                return
            d[p.val] += [c.val]
            d[c.val] += [p.val]
            dfs(c, c.left)
            dfs(c, c.right)
        
        dfs(root, root.left)
        dfs(root, root.right)
        
        q = deque()
        v = set()
        
        q += [target.val]
        v.add(target.val)
        
        level = 0
        while q:
            if level == k:
                return list(q)
            size = len(q)
            for _ in range(size):
                p = q.popleft()
                for c in d[p]:
                    if c not in v:
                        q += [c]
                        v.add(c)
            level += 1
        return []
                    
    
    '''
    Data Structure:
        1.  dfs(parent node, child node)
        	(1) base case: if child ndoe is None: return
            (2) recursion: 
            	<1> after adding  child to list of parent
            	<2> after adding parent to list of child
            	dfs(left child), dfs(right child)
        
        2.  d:
            +: during recurion of dfs, d[node] = [list of children node + parent]
            
        3.  q: 
            (1) pop: iterating nodes of a whole level
            (2) push: push d[node] when iterating nodes of a whole level 
        
        4.  v:
            +: along with pushing one node into q
            
    Algorithm:
    	1. 	convert original tree into a new one whose root is target node by dfs
    	
    	2.	find nodes of level k
    '''
        
```







## Sliding Window

### 3.Longest Substring Without Repeating Characters



```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        d = defaultdict(int)
        n = len(s)
        l = 0
        res = 0
        for r in range(n):
            d[s[r]] += 1
            while l <= r and d[s[r]] > 1:
                d[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
                
        '''
        Data Structure:
            1.  right: 
                +1: move one by one
                
            2.  left:
                +1: while d[s[right]] > 1, left + =1
                
            3.  d:
                +: if right move one, d[s[right]] += 1
                -: while d[s[right]] > 1, d[s[left]] -= 1
        '''
```



## 424 Longest Repeating Character Replacement

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = defaultdict(int)
        res = 0
        max_freq = 0
        l = 0
        for r in range(len(s)):
            ch = s[r]
            count[ch] += 1
            max_freq = max(max_freq, count[ch])
            while l <= r and r - l + 1 > max_freq + k:
                count[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
    
    '''
    Data Structure:
        1.  count: (char, freq) in sliding window
        
        2.  max_freq: max freq in count
        
        3.  right:
            (1) initialize: 0
            (2) +1: move one by one
        
        4.  left:
            (1) initialize: 0
            (2) +1: while length of sliding window > max_freq + k
    
    
    '''
```



### 1151.Minimum Swaps to Group All 1's Together

```python
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        k = sum(data)
        n = len(data)
        one = 0
        l = 0
        res = inf
        for r in range(n):
            one += data[r]
            if r >= k:
                l = r - k
                one -= data[l]
            res = min(res, k - one)
        return res
        '''
        Data Structure:
            1.  r : move right
            
            2.  one: 
                +: move one by one and add data[r]
                -: if r >= k, remove leftmost element in sliding window, namely data[r - k]
            
        Algorithm:
            if there are k 1 in total, we can try to find a sliding window of length k with max 1 in it
            1. count 1
            2. moving sliding window
        '''
```





## heap

### 253. Meeting Room II

```python
class Solution:
    from heapq import heappop, heappush
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        h = []
        for s, e in intervals:
            heappush(h, [s, 1])
            heappush(h, [e, -1])
            
        rooms = 0
        res = 0
        while h:
            _, flag = heappop(h)
            if flag > 0:
                rooms += 1
                res = max(res, rooms)
            else:
                rooms -= 1
        return res
                
        '''
        Data Structure:
            1.  h: 
                (1) pop: iterate each [timeStamp, start/end]
                (2) push: initialize h with [timeStamp, start/end]
            
            2.  rooms:
                +: when iterating h, if start, rooms += 1 ;else rooms -=1
                
        Algorithm:
            1.  initialize h
            2.  iterate h by popping out () with min timeStamp
        '''
```

#### 370. Range Addition

```python
class Solution:
    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        compact = [0] * length
        for i, j, num in updates:
            compact[i] += num
            if j + 1 <= length - 1:
                compact[j + 1] -= num
        
        cur = 0
        res = []
        for num in compact:
            cur += num
            res += [cur]
        return res
        '''
        Data Structure:
            1.  compact:
            	for each range[i, j], num
            	+: compact[i] += num
            	-: compact[j + 1] -= num
                
            2.  res:
                +: res[i] = compact[0] + ... + compact[i]
            
        Algorithm:
            1.  compact updates
            2.  iterate each point in updates to get final result
        '''
```



### 696 Count Binary Substrings

```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        p1, p2 = 1, 0
        s += '2'
        n = len(s)
        res = 0
        for i in range(1, n):
            if s[i] == s[i - 1]:
                p1 += 1
            else:
                res += min(p1, p2)
                p2 = p1
                p1 = 1
        return res
                
        '''
        Explanation:
        |000| |111|
        part2 part1
        left  right
        
        Data Structure:
            1.  part1:
                +: if s[i] == s[i - 1], part1 += 1
                -: if s[i] != s[i - 1], part1 = 1
                
            2.  part2:
                +: if s[i] != s[i - 1], part2 = part1
            
            3.  res:
                +: before switch part1 into part2, res += min(part1, part2)
        
        Algorithm:
            1.  add '2' to end of s
            2.  iterate each char in s:
                if s[i] == s[i - 1]:
                    part1 += 1
                else:
                    update res
                    switch part1 to part2
                    part1 = 1
        '''
```



### 973.K Closest Points to Origin



```python
from heapq import heappush, heappop
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        h = []
        for i, (x, y) in enumerate(points):
            d = x ** 2 + y ** 2
            heappush(h, (-d, [x, y]))
            if i >= k:
                heappop(h)
        
        res = []
        while h:
            _, [x, y] = heappop(h)
            res += [[x, y]]
        return res
        
        '''
        Data Structure:
            1.  h:
                (1) heappop: if after pushing a new point, len(h) > k, pop out the node with min dist 
        
                (2) heappush: iterate each points, push(-dist, x, y)
        '''
```





### 1167.Minimum Cost to Connect Sticks

heap + greedy

```python
from heapq import heappop, heappush
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        h = []
        res = 0
        for s in sticks:
            heappush(h, s)
            
        while len(h) >= 2:
            s1, s2 = heappop(h), heappop(h)
            res += s1 + s2
            heappush(h, s1 + s2)
        return res
    ''' 
    Data Structure:
        1.  h:
            (1) pop: while len(h) >= 2, pop out two min sticks
            (2) push: 
                <1> add all sticks into h at first from shortest to longest
                <2> after popping out two min sticks, push in their sum
    '''
```





### 1353.Maximum Number of Events That Can Be Attended 没写







## Design



### 146.LRU Cache



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
            (1) +:
            	<1> add 1 non-exited {key, value} pair
            	<2> if 1 pair exists, put it back to make it least recently used
                
            (2) -:
            	<1> exceed capacity after putting a new pair
            	<2> put a existed pair -> make it least recently used
            	<3> get a existed pair -> make it least recently used
            	
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





### 348.Design Tic-Tac-Toe



```python
class TicTacToe:
    def __init__(self, n: int):
        self.rows = [0] * n
        self.cols = [0] * n
        self.diag = 0
        self.anti_diag = 0
        self.n = n
        
    def move(self, i: int, j: int, player: int) -> int:
        self.rows[i] += (1 if player == 1 else -1)
        
        self.cols[j] += (1 if player == 1 else -1)
        
        if i == j:
            self.diag += (1 if player == 1 else -1)
        
        if i == self.n - 1 - j:
            self.anti_diag += (1 if player == 1 else -1)
        
        if abs(self.rows[i]) == self.n or abs(self.cols[j]) == self.n or abs(self.diag) == self.n or abs(self.anti_diag) == self.n:
            return player
        else:
            return 0
    '''
    Data Structure:
        1.  rows / cols / diag / antiDiag
            +1: if palyer == 1, client.move(i, j): rows[i] += 1, cols[i] += 1, diag[i] += 1 if i == j, antiDiag[i] += 1 if i == n - 1 - j
            -1: if player == -1, client.move(i, j): rows[i] += -1, cols[i] += -1, diag[i] += -1 if i == j, antiDiag[i] += -1 if i == n - 1 - j
        
    Algorithm:
        1.  move(i, j, player):
            after operations, return:
            (1) if one of r/c/d/antiD == n, reutnr 1
            (2) else, return 0 
    '''
```





#### 1275.Find Winner on a Tic Tac Toe Game

```python
class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        n = 3
        row = [0] * n
        col = [0] * n
        player = 1
        diag = antiDiag = 0
        
        for i, j in moves:
            row[i] += player
            col[j] += player
            
            if i == j:  
                diag += player
                
            if i + j == n - 1:  
                antiDiag += player
            
            if abs(row[i]) == n or abs(col[j]) == n or abs(diag) == n or abs(antiDiag) == n:
                return "A" if player == 1 else "B"
            
            player = -player
        
        return "Draw" if len(moves) == n * n else "Pending"
        '''
        Data Structure:
            1.  rows / cols / diag / antiDiag
            +1: if palyer == 1, client.move(i, j): rows[i] += 1, cols[i] += 1, diag[i] += 1 if i == j, antiDiag[i] += 1 if i == n - 1 - j
            -1: if player == -1, client.move(i, j): rows[i] += -1, cols[i] += -1, diag[i] += -1 if i == j, antiDiag[i] += -1 if i == n - 1 - j
            
            2.  player = 1 / -1 after one move
        
        Algorithm:
            iterate each move i, j
                (1) update row[i], col[j], diag, antidiag with player
                (2) if abs(row[i]) / abs(col[j]) / diag / antidiag == N 
                        return current player
                (3) change player
        '''
```





## bit



### 136.Single Number

https://leetcode.com/problems/single-number/

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
        
        """
        a ^ a == 0
        a ^ 0 == a
        """
```







## Intervals



### 56.Merge Intervals



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





### 252.Meeting Rooms



```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        if len(intervals) == 0:
            return True
        intervals.sort()
        s, e = intervals[0]
        for ns, ne in intervals[1:]:
            if s <= ns < e:
                return False
            else:
                s, e = ns, ne
        return True
        '''
        Data Structure:
            1.  s, e:
                check intersection: for new interval [ns, ne], if s <= ns < e => intersects
        
        Algorithm:
            1.  sort intervals
            2.  set s, e = intervals[0] and iterate intervals[1:]
        '''
```





```python
import heapq
# 将x压入堆中
heapq.heappush(heap, x)   

# 从堆中弹出最小的元素                                     
heapq.heappop(heap)    

# 让列表具备堆特征                                  
heapq.heapify(heap) 

# 弹出最小的元素，并将x压入堆中                                      
heapq.heapreplace(heap, x)  

# 返回iter中n个最大的元素                          
heapq.nlargest(n, iter) 

# 返回iter中n个最小的元素                                      
heapq.nsmallest(n, iter)  
```





## Greedy

### 42.Trapping Rain Water

```python
class Solution:
    def trap(self, A: List[int]) -> int:
        n = len(A)
        maxL, maxR = A[0], A[n - 1]
        l, r = 1, n - 2
        
        res = 0
        
        while l <= r:
            if maxL < maxR:
                if A[l] > maxL:
                    maxL = A[l]
                else:
                    res += maxL - A[l]
                l += 1
            else:
                if A[r] > maxR:
                    maxR = A[r]
                else:
                    res += maxR - A[r]
                r -= 1
        
        return res
        """
        Explanation:
            
            maxLeft  left   ...   right maxRight
          
            result of bars from left to right have not been computed yet
            I do not know how much water can they trap
            
            maxLeft is height of tallest bar whose idx < idx of left
            maxright is height of tallest bar whose idx > idx of right
            
            now we want to know left / right can trap how much water
            
            if maxLeft < maxRight:
                we can compute result of left
                because it is determined by smaller boundary
                and smaller boundary must be maxLeft 
                
                if left is higher than maxLeft --> there will be not water
                    --> only update maxleft and left += 1
                
                else: trap water and make current bar has same height with maxLeft
                    --> only update res(trap water) and left += 1
                    
        Data Structure:
            1. 	maxLeft and maxRight
            	(1) initialize: A[0] or A[n - 1]
            	(2) become bigger:
            		<1> maxLeft: if maxLeft is smaller and maxLeft < A[left] 
            		
            2. left and right:
            	left + 1 or right - 1: if maxLeft is smaller or maxRight is smaller
            
        Algorithm:
            1. initialize maxLeft and maxRight = nums[0] and [n - 1]
                left should be 1 and right = n - 2
                
            2.  while left <= right:
                    (1) if maxLeft < maxRight;
                            deal with left
                        <1> if height of left >= maxLeft:
                                update maxLeft
                                left += 1
                        
                        <2> if height of left < maxLeft:
                                update res
                                left += 1
                        
                    (2) else deal with right
        """
```



### 926.Flip String to Monotone Increasing

```python
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        zero, one = 0, 0
        for c in s:
            if c == '0':
                zero += 1
                
        n = len(s)
        res = n
        for c in s:
            if c == '0':
                zero -= 1
            res = min(res, one + zero)
            if c == '1':
                one += 1
        return res
        '''
        0 0 0 |pos| 1 1 1: 
         left        right
        Data Structure:
            1.  zero(on the right):
                (1) +: amount of 0 at first
                (2) -: before update res, if s[pos] == 0, zero -= 1, indicating 0 on the right
                
            2.  one(on the left)
                +: after update res, if s[pos] == 1, one += 1, indicating 1 on the left
                
            3.  res: 
                after update zero on the right, res = min(res, one + zero)
        
        Algorithm:
            1.  initialize zero = total number of '0'
            2.  iterate each char:
                (1) update zero 
                (2) update res
                (3) update one
        '''
```



### 1492.The kth Factor of n

```python
class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        bigger_factors = []
        for factor in range (1, int(sqrt(n)) + 1):
            if n % factor != 0:
                continue
            k -= 1
            if k == 0:
                return factor
            if factor ** 2 != n:
                bigger_factor = n // factor
                # bigger_factors += [bigger_factor]
                bigger_factors = [bigger_factor] + bigger_factors
        if k > len(bigger_factors):
            return -1
        return bigger_factors[k - 1]
        '''
        factors 1, 2, 4, 8, 16, n = 16
        
        smaller factors: 1, 2, 4 bigger factors: [8, 16]
        we can calculate 8, 16 from 1, 2
                
        Data Structure:
            1.  bigger_factors:
                +: if n % factor == 0 and factor ** 2 != n, add n // factor to bigger_factors 
                
            2.  k:
                -: iterate factor, k -= 1
        
        Algorithm:
            1.  Get all factors from 1 to sqrt(n)
                if k == 0 during this time, return the factor
            
            2.  if k > len(factors), no enough factors
                else: n // factork[-k]
        '''
```





### 1710.Maximum Units on a Truck

```python
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        A = sorted(boxTypes, key = lambda x: -x[1])
        k = truckSize
        res = 0
        for c, u in A:
            if c < k:
                res += c * u
                k -= c
            else:
                res += k * u
                return res
        return res
        
        """
        Data Structure:
            A: (number of box, unit of box) sorted by unit from big to small
            k:  remaining boxes to put on truck
            
        Algorithm:
            (1) sort A, initialize k = truckSize
            (2) for each box in A:
                    if number of box < k:
                        res += count * unit
                        k -= number of box
                    else: number of box >= k
                        count = k
                        res += count * unit
                        return res
            (3) return res
        """
```



### 1648.Sell Diminishing-Valued Colored Balls   没写



### 283.Move Zeroes

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        leftmost = 0
        for num in nums:
            if num != 0:
                nums[leftmost] = num
                leftmost += 1
        n = len(nums)
        for i in range(leftmost, n):
            nums[i] = 0 
        
        '''
        Data Structure:
            1.  nums:
                (1) swap: if nums[idx] != 0, swap it with nums[leftmost]
            
            2.  leftmost: idx of leftmost 0
                +: after swapping nums[idx] into nums[leftmost], leftmost += 1
        
        Algorithm:
            1.  move all non-zero nums as left as possible
            2.  set all other nums (from leftmost to end) to zero
        '''
```



## Sort

### 937.Reorder Data in Log Files

```python
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        letter, digit = [], []
        for log in logs:
            if log.split()[1].isdigit():
                digit += [log]
            else:
                letter += [log]
        
        letter.sort(key=lambda x: (x.split()[1:], x.split()[0]))
        return letter + digit
        
        '''
        Explanation:
            1. identifier, content = first element, array of remaining elements
            2. letter-logs, digit-logs: content[0] can determine
        
        Data Structure:
            1.  letter-logs:
                (1) sort by content: x.split()[1:]
                (2) sort by identifier: x.split()[0]
                
        Algorithm:
            1.  divide logs into letter-logs and digit-logs
            2.  sort letter log
            3.  attach digit logs to end of letter logs
        '''
```



# Amazon OA

## 78 subset

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for num in nums:
            newArr = []
            for item in res:
                newArr.append(item + [num])
            res += newArr
        return res
```



## maxima frequency of string

https://www.1point3acres.com/bbs/thread-917933-1-1.html

```python
def find_max_freq(s):
    n = len(s)
    count = defaultdict(int)
    max_count = defaultdict(int)
    max_freq = 0
    max_freq_chs = set()
    for i in range(n):
        count[ch] += 1
        if count[ch] == max_freq:
            max_freq_chs.add(ch)
        elif count[ch] > max_freq:
            max_freq_chs = set()
            max_freq_chs.add(ch)
			max_freq = max(max_freq, count[ch])
   		
        for max_ch in max_freq_chs:
            max_count[max_ch] += 1	
	
    res = 0
    for ch in max_count:
        if max_count[ch] > res:
            res = max(res, max_count)
  	return res
        
```



## Amazon parcles maximum of middle box

```python
def findMaximumMiddleBox(capacities):
    res = -1
    seen = set()
    has_factor = set()
    capacities.sort()
    for a in capacities:
        for factor in range(1, int(sqrt(a)) + 1):
            if a % factor != 0:
                continue
            if factor in seen:
                has_factor.add(a)
                if factor in has_factor:
                    res = max(res, factor)
		seen.add(a)
  	return res
```



## 424 character Replacement

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = defaultdict(int)
        res = 0
        max_freq = 0
        l = 0
        for r in range(len(s)):
            ch = s[r]
            count[ch] += 1
            max_freq = max(max_freq, count[ch])
            while l <= r and r - l + 1 > max_freq + k:
                count[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
    
    '''
    Data Structure:
        1.  count: (char, freq) in sliding window
        
        2.  max_freq: max freq in count
        
        3.  right:
            (1) initialize: 0
            (2) +1: move one by one
        
        4.  left:
            (1) initialize: 0
            (2) +1: while length of sliding window > max_freq + k
    '''
```



## 659 if subsequence

```python
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        remain = collections.Counter(nums)
        end = defaultdict(int)
        for num in nums:
            if remain[num] <= 0:
                continue
            remain[num] -= 1
            if end[num - 1] > 0:
                end[num - 1] -= 1
                end[num] += 1
            elif remain[num + 1] > 0 and remain[num + 2] > 0:
                remain[num + 1] -= 1
                remain[num + 2] -= 1
                end[num + 2] += 1
            else:
                return False
        return True
        '''
        Data Structure:
            1.  remain: 
                remain[a]: # of remaining num a
                +:  initialize
                -:  (1) a can serve as last num of a subsequence
                    (2) a can start a new subsequence or serve as a new end
            
            2.  end:
                end[a]: # of subsequences whose last num is a
                +:  (1) a - 2 starts a new subsequence and a is last num of that subsequence
                    (2) a can serve as new end
                -:  a + 1 can serve as a new subsequence
        
        Algorithm:
            1.  initialize a remain and end
            2.  for each num:
                (1) check if its freq >= 1
                (2) check if it can become new end of an existing subsequence
                (3) check if it can serve as a new start
        ''' 
```



## 719 k-smallest pairs

```python
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        right = nums[-1] - nums[0]
        while left < right:
            mid = left + (right - left) // 2
            if self.numSmallerPairs(nums, mid) >= k:
                right = mid
            else:
                left = mid + 1
        return left
    
    def numSmallerPairs(self, nums: List[int], dist: int) :
        n = len(nums)
        count = 0
        left = 0
        right = 1
        while right < n:
            if nums[right] - nums[left] <= dist:
                count += right - left
                right += 1
            else:
                left += 1
        return count
```



## 2214 min health

```python
class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        max_damage = max(damage)
        min_health = sum(damage) - min(max_damage, armor) + 1
        
        return min_health
```



## 2281 wizzard

```python
class Solution:
    def totalStrength(self, A: List[int]) -> int:
        res = 0
        n = len(A)
        preSum = [0]
        curSum = 0
        stack = [-1]
        A += [0]
        for r, a in enumerate(A):
            curSum += a
            preSum += [curSum + preSum[-1]]
            while stack and A[stack[-1]] > a:
                i = stack.pop()
                l = stack[-1]
                ln = i - l
                rn = r - i
                left_sum = (preSum[i] - preSum[max(l, 0)]) * rn
                right_sum = (preSum[r] - preSum[i]) * ln
                min_strength = A[i]
                res += min_strength * (right_sum - left_sum)
            stack += [r]
        return res % (10 ** 9 + 7)
                
        '''
        Data Strcuture:
            1.  preSum:
                sum of A[i] ~ A[j]: preSum[j] - preSum[i]
            
            2.  stack: [left boundary, min]
                (1) +:  after popping out min, add cur num to stack
                (2) -:  while stack[-1] < A[i]: update res
        
        '''
```

