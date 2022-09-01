package main

import (
	"fmt"
)

/*
next string: s1 serves as left boundary, s2 servers as right boundary, if s[i] == s1[i], left boundary will be invalid
	same works for right boundary

evil string: each match, idx + 1, if idx == len(evil), we get a false string

dp: 4d memo -> key / bitmask

*/

func findGoodStrings(n int, s1 string, s2 string, evil string) int {
	dp := map[int]int{}
	res := dfs1397(0, n, 0, len(evil), "", s1, s2, evil, true, true, dp)
	return res
}

func dfs1397(curIdx, n, curEvilIdx, evilLen int, curS, s1, s2, evil string, isLeftBoundary, isRightBoundary bool, dp map[int]int) int {
	// fmt.Println(curS)
	// 1. Base Case
	// base case 1 --> evil match
	if curEvilIdx == evilLen {
		return 0
	}
	// base case 2 --> good string
	if curIdx == n {
		// fmt.Println(curS, IsSmaller(s1, curS), IsSmaller(curS, s2))
		return 1
	}
	// 2. Check dp
	key := GetKey(curIdx, curEvilIdx, isLeftBoundary, isRightBoundary)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// 3. Next Recursion
	res := 0
	// Get Next Char by isLeftBoundary and isRightBoundary
	start, end := 0, 25
	if isLeftBoundary {
		start = int(s1[curIdx] - 'a')
	}
	if isRightBoundary {
		end = int(s2[curIdx] - 'a')
	}
	for i := start; i <= end; i++ {
		c := byte('a' + i)
		newS := curS + string(c)
		// check whether new string contains evil string
		var newCurEvilIdx int
		if c == evil[curEvilIdx] {
			newCurEvilIdx = curEvilIdx + 1
		} else {
			newCurEvilIdx = 0
		}
		newIsLeftBoundary := isLeftBoundary && (c == s1[curIdx])
		newIsRightBoundary := isRightBoundary && (c == s2[curIdx])
		res += dfs1397(curIdx+1, n, newCurEvilIdx, evilLen, newS, s1, s2, evil, newIsLeftBoundary, newIsRightBoundary, dp)
		res %= 1000000007
	}
	fmt.Println(res)
	// 4. Update dp
	dp[key] = res
	return res
}

func GetKey(curIdx, evilIdx int, isLeftBoundary, isRightBoundary bool) int {
	// max n is 500 --> 10 bit
	// max length of evil -> 50 -> 6 bit
	// isLeftboundary -> 1 bit
	// isRightBoundary -> i bit
	// key: curIdx + evilIdx, isLeft, isRight
	key := 0
	if isRightBoundary {
		key += 1
	}
	if isLeftBoundary {
		key += (1 << 1)
	}
	key += (evilIdx << 2)
	key += (curIdx << 8)
	return key
}

func main() {
	n := 2
	s1 := "aa"
	s2 := "da"
	evil := "b"
	res := findGoodStrings(n, s1, s2, evil)
	fmt.Println(res)
}
