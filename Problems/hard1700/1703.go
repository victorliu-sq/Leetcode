package main

import (
	"fmt"
	"math"
)

/*
	1. 	Record idx of all 1s in nums
	2.	For each consecutive k 1s: try to move 1 to median idx
	3.	(1) [1, 2, 3, 4, 5, 6] moves to median:
			(4 - 1) + (4 - 2) + (4 - 3) + (4 - 4) + (5 - 4) + (6 - 4)
			can be simplified to (4 + 5 + 6) - (1 + 2 + 3) -> preSum1 - preSum2
		(2)	[1, 2, 3, 4, 5, 6] differs from [4, 4, 4, 4, 4, 4] --> [3, 2, 1, 0, 1, 2] --> radius = 3 -->  (radius + 1) * radius - (radius + 1)
			[2, 3, 4, 5, 6] differs from [4, 4, 4, 4, 4] --> [2,1,0,1,2] --> radius = 2 -->  radius * (radius - 1)

*/

func minMoves(nums []int, k int) int {
	indices := []int{}
	// 1. Record indices of all 1s int num
	for i, num := range nums {
		if num == 0 {
			continue
		}
		indices = append(indices, i)
	}
	fmt.Println(indices)
	// 2. Get preSum of indices
	n := len(indices)
	preSum := make([]int, n+1)
	for i := 0; i < n; i++ {
		preSum[i+1] = indices[i] + preSum[i]
	}
	// 3. For each consecutive k 1s: try to move 1 to median idx
	radius := (k + 1) / 2
	res := math.MaxInt32
	for i := 0; i <= n-k; i++ {
		j := i + k - 1
		A := preSum[j+1] - preSum[j+1-radius]
		B := preSum[i+radius] - preSum[i]
		res = min1703(res, A-B)
		fmt.Println(A, B)
	}
	var cost int
	if k%2 == 1 {
		cost = (radius - 1) * radius
	} else {
		cost = (radius+1)*radius - radius
	}
	fmt.Println(cost)
	return res - cost
}

func min1703(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	nums := []int{1, 0, 0, 0, 0, 0, 1, 1}
	k := 3
	res := minMoves(nums, k)
	fmt.Println(res)
}
