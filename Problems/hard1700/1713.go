package main

import (
	"fmt"
	"math"
)

/*
LIS: 300 -> dp + binary search
LCS: dp

we see that target contains no duplicates.
Therefore, the value and index of the elements in target have an one-to-one mapping relationship.
Using this infomaiton, we can map all the values in arr to their corresponding index in target.
If a value is not existed in target, we just skip it.
*/

func minOperations(target []int, arr []int) int {
	// Get a hashmap [num, idx] for target
	num2idx := map[int]int{}
	for idx, num := range target {
		num2idx[num] = idx
	}
	fmt.Println(num2idx)
	indices := []int{}
	// convert arr into an array of indices
	for _, num := range arr {
		if _, ok := num2idx[num]; ok {
			indices = append(indices, num2idx[num])
		}
	}
	fmt.Println(indices)
	// find LIS of indices
	return len(target) - LIS(indices)
}

func LIS(nums []int) int {
	size := 0
	n := len(nums)
	dpLIS := make([]int, n+1)
	for i := 1; i < n; i++ {
		dpLIS[i] = math.MaxInt32
	}
	for _, num := range nums {
		idx := BinarySearchLastSmaller(dpLIS, num, size)
		dpLIS[idx+1] = min1713(dpLIS[idx+1], num)
		size = max1713(size, idx+1)
	}
	return size
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

func max1713(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func min1713(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	// target := []int{5, 1, 3}
	// arr := []int{9, 4, 2, 3, 4}
	target := []int{6, 4, 8, 1, 3, 2}
	arr := []int{4, 7, 6, 2, 3, 8, 6, 1}
	res := minOperations(target, arr)
	fmt.Println(res)
}
