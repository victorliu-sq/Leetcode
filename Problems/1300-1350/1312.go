package main

import "fmt"

/*
(1) mbabdm -> part1:s part2: Reversed(s)
(2) dp[i][j]: max # of matched letters between part1 & part2
*/
func minInsertions(s string) int {
	m := len(s)
	part1 := s
	part2 := Reverse(s)
	fmt.Println(part1)
	fmt.Println(part2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, m+1)
	}
	maxMatched := 0
	for i := 1; i <= m; i++ {
		for j := 1; j <= m; j++ {
			if part1[i-1] == part2[j-1] {
				// fmt.Println(i, j)
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
			maxMatched = max(maxMatched, dp[i][j])
			// fmt.Println(maxMatched)
		}
	}
	return m - maxMatched
}

func Reverse(s string) string {
	var res string
	for _, c := range s {
		res = string(c) + res
	}
	return res
}

func max(n1 int, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	s := "mbabdm"
	res := minInsertions(s)
	fmt.Println(res)
}
