package main

import (
	"fmt"
)

/*
Longest path problem : dfs + dp
*/

func maxJumps(arr []int, d int) int {
	nextIndices := GetNextIndices(arr, d)
	res := 1
	dp := newDP1340(len(arr))
	// for each start, try to get longest path
	for start := 0; start < len(arr); start++ {
		temp := dfs1340(start, nextIndices, &dp)
		if temp > res {
			res = temp
		}
	}
	return res
}

func dfs1340(curIdx int, nextIndices map[int][]int, dp *[]int) int {
	// base case
	if len(nextIndices[curIdx]) == 0 {
		return 1
	}
	// check dp
	if (*dp)[curIdx] != -1 {
		return (*dp)[curIdx]
	}
	// recursion
	res := 1
	for _, nextIdx := range nextIndices[curIdx] {
		temp := 1 + dfs1340(nextIdx, nextIndices, dp)
		if temp > res {
			res = temp
		}
	}
	// update dp
	(*dp)[curIdx] = res
	return res
}

func GetNextIndices(arr []int, d int) map[int][]int {
	n := len(arr)
	nextIndices := map[int][]int{}
	for i := 0; i < n; i++ {
		for j := i + 1; j <= i+d; j++ {
			if j < n && arr[i] > arr[j] {
				nextIndices[i] = append(nextIndices[i], j)
			} else {
				break
			}
		}
		for j := i - 1; j >= i-d; j-- {
			if j >= 0 && arr[i] > arr[j] {
				nextIndices[i] = append(nextIndices[i], j)
			} else {
				break
			}
		}
	}
	return nextIndices
}

func newDP1340(n int) []int {
	dp := make([]int, n)
	for i := range dp {
		dp[i] = -1
	}
	return dp
}

func main() {
	arr := []int{6, 4, 14, 6, 8, 13, 9, 7, 10, 6, 12}
	d := 2
	res := maxJumps(arr, d)
	fmt.Println(res)
}
