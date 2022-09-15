package main

import (
	"fmt"
	"math"
	"sort"
)

/*
312, 1000, 1039
*/

func minCost(n int, cuts []int) int {
	stones := []int{}
	sort.Slice(cuts, func(i, j int) bool {
		if cuts[i] <= cuts[j] {
			return true
		}
		return false
	})
	prev := 0
	cuts = append(cuts, n)
	for _, cut := range cuts {
		stones = append(stones, cut-prev)
		prev = cut
	}
	// fmt.Println(stones)
	res := mergeStones(stones, 2)
	return res
}

// Merge stones
func mergeStones(stones []int, m int) int {
	n := len(stones)
	preSum := make([]int, n+1)
	for i := 1; i <= n; i++ {
		preSum[i] = preSum[i-1] + stones[i-1]
	}
	// fmt.Println(preSum)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	for d := m; d <= n; d++ {
		for i := 0; i+d-1 <= n-1; i++ {
			j := i + d - 1
			dp[i][j] = math.MaxInt32
			// try to merge all left stones into one stone
			// 1 -> 1 + (k - 1) -> 1 + (k - 1) + (k - 1)
			for k := i; k < j; k += m - 1 {
				dp[i][j] = min1547(dp[i][j], dp[i][k]+dp[k+1][j])
			}
			// fmt.Println(dp[i][j])
			// check whether all stones can be merged together
			if (j-i)%(m-1) == 0 {
				dp[i][j] += preSum[j+1] - preSum[i]
			}
			// fmt.Println(i, j, dp[i][j])
		}
	}
	return dp[0][n-1]
}

func min1547(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	// stones := []int{3, 2, 4, 1}
	// k := 2
	// res := mergeStones(stones, k)
	// fmt.Println(res)

	n := 7
	cuts := []int{1, 3, 4, 5}
	res := minCost(n, cuts)
	fmt.Println(res)
}
