package main

import "fmt"

func minDays(n int) int {
	dp := map[int]int{}
	res := dfs1553(n, dp)
	return res
}

func dfs1553(n int, dp map[int]int) int {
	// base case
	if n == 0 {
		return 0
	}
	// check dp
	if _, ok := dp[n]; ok {
		return dp[n]
	}
	// next recursion
	res := n
	/*
		// divisible by 2
		if n%2 == 0 && n/2 > 0 {
			eat := n / 2
			temp := 1 + dfs1553(n-eat, dp)
			res = min1553(res, temp)
		}
		// divisible by 3
		if n%3 == 0 && 2*(n/3) > 0 {
			eat := 2 * (n / 3)
			temp := 1 + dfs1553(n-eat, dp)
			res = min1553(res, temp)
		}
		// eat 1
		{
			eat := 1
			temp := 1 + dfs1553(n-eat, dp)
			res = min1553(res, temp)
		}
	*/
	// try to be divisible by 2
	{
		nDiv2 := n - n%2
		eat := nDiv2 / 2
		// n % 2 days to be divisible by 2, 1 day to eat n / 2
		temp := 1 + n%2 + dfs1553(nDiv2-eat, dp)
		res = min1553(res, temp)
	}
	// try to be divisible by 3
	{
		nDiv2 := n - n%3
		eat := 2 * (nDiv2 / 3)
		// n % 3 days to be divisible by 2, 1 day to eat 2 * (n / 3)
		temp := 1 + n%3 + dfs1553(nDiv2-eat, dp)
		res = min1553(res, temp)
	}
	dp[n] = res
	return res
}

func min1553(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	n := 10
	res := minDays(n)
	fmt.Println(res)
}
