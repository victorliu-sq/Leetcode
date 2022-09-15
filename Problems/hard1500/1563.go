package main

import "fmt"

func stoneGameV(stones []int) int {
	n := len(stones)
	preSum := make([]int, n+1)
	for i := 1; i <= n; i++ {
		preSum[i] = preSum[i-1] + stones[i-1]
	}
	fmt.Println(preSum)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	// len == 2
	// for i := 0; i < n-1; i++ {
	// 	dp[i][i+1] = min1563(stones[i], stones[i+1])
	// }
	// len >= 2
	for d := 2; d <= n; d++ {
		for i := 0; i+d-1 <= n-1; i++ {
			j := i + d - 1
			dp[i][j] = 0
			// divide [i, j] into 2 halves [i, k][k + 1, j]
			for k := i; k <= j-1; k++ {
				sum1, sum2 := preSum[k+1]-preSum[i], preSum[j+1]-preSum[k+1]
				if sum1 < sum2 {
					dp[i][j] = max1563(dp[i][j], sum1+dp[i][k])
				} else if sum1 > sum2 {
					dp[i][j] = max1563(dp[i][j], sum2+dp[k+1][j])
				} else {
					dp[i][j] = max1563(dp[i][j], max1563(sum1+dp[i][k], sum2+dp[k+1][j]))
				}
			}
			// fmt.Println(i, j, dp[i][j])
		}
	}
	return dp[0][n-1]
}

func max1563(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func min1563(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	stones := []int{6, 2, 3, 4, 5, 5}
	res := stoneGameV(stones)
	fmt.Println(res)
}
