package main

import "fmt"

/*
dp[i]: max value of (Alice score - Bob score) from stone[i] to stone[n - 1]
		[s1, s2, s3, s4 ,s5] if we choose s1, s2, s3
		dp[i] -> sum of [s1, s2, s3] - dp[4]
*/

func stoneGameIII(stoneValue []int) string {
	n := len(stoneValue)
	dp := make([]int, n)
	for i := n - 1; i >= 0; i-- {
		total := 0
		dp[i] = -100000
		for j := 1; j <= 3 && i+j-1 < n; j++ {
			total += stoneValue[i+j-1]
			// dp[i] = total - dp[i+j]
			var temp int
			if i+j <= n-1 {
				temp = dp[i+j]
			}
			dp[i] = max1406(dp[i], total-temp)
			// fmt.Println(i, j)
		}
	}
	if dp[0] > 0 {
		return "Alice"
	} else if dp[0] < 0 {
		return "Bob"
	}
	return "Tie"
}

func max1406(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	stones := []int{1, 2, 3, 7}
	res := stoneGameIII(stones)
	fmt.Println(res)
}
