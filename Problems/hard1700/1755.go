package main

import (
	"fmt"
	"math"
	"sort"
)

/*
	1. Generate all subsequences of nums
	2. Find the subsequence whose sum is most closest to goal
*/

func minAbsDifference(nums []int, goal int) int {
	sums := []int{}
	dfs1755(0, nums, []int{}, &sums)
	// sort sums
	sort.Ints(sums)
	// fmt.Println(sums)
	// find closest sum
	cloest := BinarySearchCloest(sums, goal)
	return GetDifference(cloest, goal)
}

func BinarySearchCloest(sums []int, target int) int {
	candidate := sums[0]
	left, right := 0, len(sums)-1
	for left <= right {
		mid := left + (right-left)/2
		sum := sums[mid]
		if sum == target {
			return target
		}
		if GetDifference(candidate, target) > GetDifference(sum, target) {
			candidate = sum
		}
		if sum < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return candidate
}

func GetDifference(n1, n2 int) int {
	return int(math.Abs(float64(n1) - float64(n2)))
}

func dfs1755(curIdx int, nums, path []int, res *[]int) {
	n := len(nums)
	if curIdx == n {
		sum := GetSum(path)
		// fmt.Println(path)
		*res = append(*res, sum)
		return
	}
	dfs1755(curIdx+1, nums, path, res)
	dfs1755(curIdx+1, nums, append(path, nums[curIdx]), res)
}

func GetSum(nums []int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	return sum
}

func main() {
	nums := []int{7, -9, 15, -2}
	goal := -5
	res := minAbsDifference(nums, goal)
	fmt.Println(res)
}
