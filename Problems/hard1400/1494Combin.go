package main

import "fmt"

func GetCombinations(nums []int, k int) [][]int {
	combinations := [][]int{}
	dfsCombination(0, k, []int{}, nums, &combinations)
	return combinations
}

func dfsCombination(start, k int, curNums, nums []int, combiantions *[][]int) {
	if len(curNums) == k {
		// fmt.Println(curNums)
		*combiantions = append(*combiantions, curNums)
		return
	}
	// not enough nums
	n := len(nums)
	if len(curNums)+n-1-start+1 < k {
		return
	}
	end := n - 1
	for i := start; i <= end; i++ {
		newNums := CopySlice1494(curNums)
		newNums = append(newNums, nums[i])
		// fmt.Println("newNums", newNums)
		dfsCombination(i+1, k, newNums, nums, combiantions)
	}
}

func CopySlice1494(arr []int) []int {
	res := []int{}
	for _, num := range arr {
		res = append(res, num)
	}
	return res
}

func main() {
	// nums := []int{1, 2, 3, 4, 5}
	// k := 3
	nums := []int{2, 3}
	k := 2
	res := GetCombinations(nums, k)
	fmt.Println(res)
}
