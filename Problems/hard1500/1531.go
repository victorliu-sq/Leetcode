package main

import (
	"fmt"
	"strconv"
)

/*
Algorithm development:
(1) dfs compression w/o  deletion
(2) deletion w/o dp
(3) dfs + dp

for each char, decide whether to delete this char
*/

func getLengthOfOptimalCompression(s string, k int) int {
	dp := map[int]int{}
	res := dfs1531(0, k, s, "", '-', 0, dp)
	return res
}

func dfs1531(curIdx, k int, s, curS string, lastChar byte, lastCharCount int, dp map[int]int) int {
	n := len(s)
	// base case
	if curIdx == n {
		newS := curS + string(lastChar)
		if lastCharCount > 1 {
			newS += strconv.Itoa(lastCharCount)
		}
		fmt.Println(newS)
		return 0
	}
	// check dp
	key := GetKey(curIdx, k, lastCharCount, lastChar)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// next recursion
	res := n
	// (1) delete cur curChar
	if k > 0 {
		temp := dfs1531(curIdx+1, k-1, s, curS, lastChar, lastCharCount, dp)
		res = min1531(res, temp)
	}
	// (2) do not delete curChar
	var newCharCount int
	newS := curS
	var inc int
	newChar := s[curIdx]
	if lastChar == newChar {
		// if same
		newChar = lastChar
		newCharCount = lastCharCount + 1
		if lastCharCount == 1 || lastCharCount == 9 || lastCharCount == 99 {
			inc = 1
		}
	} else {
		// if different
		newChar = s[curIdx]
		newCharCount = 1
		newS += string(lastChar)
		inc = 1
		if lastCharCount > 1 {
			newS += strconv.Itoa(lastCharCount)
		}
		fmt.Println(newS)
	}
	// fmt.Println(newS)
	temp := inc + dfs1531(curIdx+1, k, s, newS, newChar, newCharCount, dp)
	res = min1531(res, temp)
	dp[key] = res
	return res
}

func min1531(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func GetKey(curIdx, k, lastCharCount int, lastChar byte) int {
	// curIdx -> 100 -> 7
	// k -> 100 -> 7
	// lastCharCount -> 100 -> 7
	// lastChar -> byte -> 8
	key := 0
	key += curIdx + k<<7 + lastCharCount<<14 + int(lastChar)<<21
	return key
}

func main() {
	s := "aaabcccd"
	k := 2
	res := getLengthOfOptimalCompression(s, k)
	fmt.Println(res)
}
