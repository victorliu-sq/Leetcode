package main

import "fmt"

func kthSmallestPath(destination []int, k int) string {
	m, n := destination[0], destination[1]
	dp := GetPathNumDp(m, n)
	fmt.Println(dp)
	curS, i, j := "", 0, 0
	curNum := 0
	for len(curS) < m+n {
		// check if we can only move horizontally or vertically
		fmt.Println(curS)
		if i == m {
			for j <= n-1 {
				curS += "H"
				j += 1
			}
			break
		}
		if j == n {
			for i <= m-1 {
				curS += "V"
				i += 1
			}
			break
		}
		// If we move Vertically, we will go over dp[i][j + 1] nums
		fmt.Println(curNum+dp[i][j+1], k)
		if k <= curNum+dp[i][j+1] {
			// add "H": 1 ~ dp[i][j + 1]
			curS += "H"
			j += 1
		} else {
			// add "V": dp[i][j + 1] + 1 ~ dp[i][j]
			curS += "V"
			curNum += dp[i][j+1]
			i += 1
		}
	}
	return curS
}

func GetPathNumDp(m, n int) [][]int {
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := m; i >= 0; i-- {
		for j := n; j >= 0; j-- {
			if i == m && j == n {
				dp[i][j] = 1
			} else if i == m {
				dp[i][j] = dp[i][j+1]
			} else if j == n {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1] + dp[i+1][j]
			}
		}
	}
	return dp
}

func main() {
	destination := []int{2, 3}
	k := 2
	res := kthSmallestPath(destination, k)
	fmt.Println(res)
}
