package main

import "fmt"

func countOrders(n int) int {
	dp := NewDP1359(n)
	return dfs1359(0, n, 0, 0, dp)
}

func dfs1359(curIdx int, n int, bitmaskP int, bitmaskD int, dp map[int]map[int]int) int {
	// base case
	if curIdx == 2*n {
		return 1
	}
	// check dp
	if _, ok := dp[bitmaskP]; ok {
		if _, ok := dp[bitmaskP][bitmaskD]; ok {
			return dp[bitmaskP][bitmaskD]
		}
	} else {
		dp[bitmaskP] = map[int]int{}
	}
	res := 0
	// next recursion
	for i := 0; i < n; i++ {
		// P not used
		if bitmaskP&(1<<i) == 0 {
			res += dfs1359(curIdx+1, n, bitmaskP|(1<<i), bitmaskD, dp)
		}
		// P used but D not used
		if bitmaskP&(1<<i) != 0 && bitmaskD&(1<<i) == 0 {
			res += dfs1359(curIdx+1, n, bitmaskP, bitmaskD|(1<<i), dp)
		}
	}
	// update dp
	dp[bitmaskP][bitmaskD] = res
	return res
}

func NewDP1359(n int) map[int]map[int]int {
	dp := map[int]map[int]int{}
	return dp
}

func main() {
	n := 2
	res := countOrders(n)
	fmt.Println(res)
}
