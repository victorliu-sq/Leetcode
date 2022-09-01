package main

import "fmt"

/*
General Idea: cut array into d subarrays
	[XXX START .... END]
	old	 new
all XXX are in curD - 1 arrays
all nums between start and end can be assigned into 1 new subarray

store temp in dp[i][j] -> i : curD, j : start

time complexity: d * n
space complexity : max(d * n, d(depth of tree))
*/

func minDifficulty(jobDifficulty []int, d int) int {
	if len(jobDifficulty) < d {
		return -1
	}
	n := len(jobDifficulty)
	dp := newDP(d+1, n)
	return dfs1335(1, d, 0, jobDifficulty, &dp, []int{})
}

func dfs1335(curD int, d int, start int, jobDifficulty []int, dp *[][]int, path []int) int {
	n := len(jobDifficulty)
	// base case
	if curD == d+1 || start == n {
		// fmt.Println(path)
		if curD == d+1 && start == n {
			return 0
		}
		return 1000000
	}
	// dp
	if (*dp)[curD][start] != -1 {
		return (*dp)[curD][start]
	}
	// recursion next: assign some jobs to one 1 day
	res := 1000000
	for end := start; end < n; end++ {
		// fmt.Println("start", start, "end", end)
		newPath := append(path, end)
		temp := GetMax(jobDifficulty[start:end+1]) + dfs1335(curD+1, d, end+1, jobDifficulty, dp, newPath)
		if temp < res {
			res = temp
		}
	}
	(*dp)[curD][start] = res
	// fmt.Println(res, dp)
	return res
}

func GetMax(arr []int) int {
	max := -1
	for i := range arr {
		if arr[i] > max {
			max = arr[i]
		}
	}
	return max
}

func newDP(m, n int) [][]int {
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
		for j := range dp[i] {
			dp[i][j] = -1
		}
	}
	return dp
}

func main() {
	jobDifficulty := []int{6, 5, 4, 3, 2, 1}
	d := 2
	res := minDifficulty(jobDifficulty, d)
	fmt.Println(res)
}
