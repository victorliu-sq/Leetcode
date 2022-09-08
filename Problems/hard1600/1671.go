package main

import (
	"fmt"
	"math"
)

func minimumMountainRemovals(nums []int) int {
	n := len(nums)
	// fmt.Println("LIS on the left")
	dp1 := make([]int, n)
	dpLIS := GetDpLIS(n)
	size := 0
	for i := 0; i < n; i++ {
		num := nums[i]
		idx := BinarySearchLastSmaller(dpLIS, num, size)
		dpLIS[idx+1] = min1671(dpLIS[idx+1], num)
		size = max1671(size, idx+1)
		dp1[i] = size
		// fmt.Println(size, dpLIS)
	}

	// fmt.Println("LIS on the right")
	dp2 := make([]int, n)
	dpLIS = GetDpLIS(n)
	size = 0
	for i := n - 1; i >= 0; i-- {
		num := nums[i]
		idx := BinarySearchLastSmaller(dpLIS, num, size)
		dpLIS[idx+1] = min1671(dpLIS[idx+1], num)
		size = max1671(size, idx+1)
		dp2[i] = size
		// fmt.Println(size, dpLIS)
	}
	// fmt.Println("dp1", dp1)
	// fmt.Println("dp2", dp2)
	res := 0
	for i := 0; i < n; i++ {
		// Notice that both dp1[i] and dp2[i] should be bigger than 2
		if dp1[i] <= 1 || dp2[i] <= 1 {
			continue
		}
		temp := dp1[i] + dp2[i] - 1
		if temp > res {
			res = temp
		}
	}
	return n - res
}

func BinarySearchLastSmaller(dpLIS []int, num int, size int) int {
	l, r := 0, size
	// fmt.Println(l, r)
	for l < r {
		m := l + (r-l)/2 + 1
		if dpLIS[m] < num {
			l = m
		} else {
			r = m - 1
		}
	}
	return l
}

func GetDpLIS(n int) []int {
	dpLIS := make([]int, n)
	dpLIS[0] = 0
	for i := 1; i < n; i++ {
		dpLIS[i] = math.MaxInt32
	}
	return dpLIS
}

func min1671(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func max1671(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	nums := []int{2, 1, 1, 5, 6, 2, 3, 1}
	res := minimumMountainRemovals(nums)
	fmt.Println(res)
}
