package main

import (
	"fmt"
	"strconv"
)

/*
If we use dfs + memo, the size of memo will be too large to allocate --> 2 ^ 108 if bitmask so we need to consider other methods

Since the cost needs to fit target, this is a knapsack problem

*/

func largestNumber(costs []int, target int) string {
	dp := make([]string, target+1)
	for i := range dp {
		dp[i] = "-"
	}
	// set "-" as invalid costs
	dp[0] = ""
	for sum := 1; sum <= target; sum++ {
		for idx, cost := range costs {
			if sum-cost >= 0 && dp[sum-cost] != "-" {
				num := strconv.Itoa(idx + 1)
				// 2 + 777 -> (1) 7772 (2) 2777
				temp1 := dp[sum-cost] + num
				temp2 := num + dp[sum-cost]
				// compare two strings
				temp := max1449(temp1, temp2)
				// fmt.Println(temp1, temp2)
				if dp[sum] == "-" {
					dp[sum] = temp
				} else {
					dp[sum] = max1449(dp[sum], temp)
				}
			}
		}
		fmt.Println(sum, dp[sum])
	}
	if dp[target] == "-" {
		return "0"
	}
	return dp[target]
}

func max1449(n1, n2 string) string {
	// (1) compare length
	if len(n1) > len(n2) {
		return n1
	} else if len(n1) < len(n2) {
		return n2
	} else {
		// (2) compare from highest bit to lowest bit
		length := len(n1)
		for i := 0; i < length; i++ {
			c1, c2 := n1[i], n2[i]
			d1, d2 := int(c1), int(c2)
			// fmt.Println(string(c1), d1, string(c2), d2)
			if d1 > d2 {
				return n1
			} else if d1 < d2 {
				return n2
			}
		}
		return n1
	}
}

func main() {
	// cost := []int{4, 3, 2, 5, 6, 7, 2, 5, 5}
	// target := 9
	cost := []int{70, 84, 55, 63, 74, 44, 27, 76, 34}
	target := 659
	res := largestNumber(cost, target)
	fmt.Println(res)
}
