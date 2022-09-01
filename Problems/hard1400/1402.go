package main

import (
	"fmt"
	"sort"
)

/*
accumulation of sum : 1 * x1 + 2 * x2 + 3 * x3 = sum1 + sum2 + sum3

to maximize the res, try to add from biggest num, if sum > 0, add it to res
*/

func maxSatisfaction(satisfaction []int) int {
	sort.Slice(satisfaction, func(i, j int) bool {
		if satisfaction[i] >= satisfaction[j] {
			return true
		}
		return false
	})
	fmt.Println(satisfaction)
	res := 0
	total := 0
	for _, s := range satisfaction {
		total += s
		if total > 0 {
			res += total
		}
	}
	return res
}

func main() {
	satisfaction := []int{-1, -8, 0, 5, -9}
	res := maxSatisfaction(satisfaction)
	fmt.Println(res)
}
