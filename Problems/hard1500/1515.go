package main

import "fmt"

/*
	1. divide array into several subarrays, if num[i] < num[i + 1], num[i + 1] is in another subarray

	2. cost of cur subarray = max of cur subarray - min of last subarray

	[3,1,5,4,2,3,2,4,2] -> [3,1] [5,4,2] [3,2] [4,2] -> 3 + (5 -1) + (3 -2) + (4 -2)
	when we move to a new subarray --> add nums[i] - num[i - 1] to cost
*/

func minNumberOperations(target []int) int {
	n := len(target)
	res := target[0]
	for i := 1; i < n; i++ {
		if target[i]-target[i-1] > 0 {
			res += target[i] - target[i-1]
		}
	}
	return res
}

func main() {
	target := []int{3, 1, 1, 2}
	res := minNumberOperations(target)
	fmt.Println(res)
}
