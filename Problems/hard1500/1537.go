package main

import "fmt"

/*
	1. All common elements must be added to res
	2. Two arrays can be divided into parts of same number
	3. Each part we can choose the bigger one
*/

func maxSum(nums1 []int, nums2 []int) int {
	// Get common elements of two arrays
	common := GetCommon(nums1, nums2)
	fmt.Println(common)
	res := 0
	for num, _ := range common {
		res += num
	}
	// Get sum of elements between any two common elements
	sums1 := GetMiddleSum(nums1, common)
	sums2 := GetMiddleSum(nums2, common)
	fmt.Println(sums1, sums2)
	// choose the bigger middle sum and add it to res
	for i := 0; i < len(sums1); i++ {
		sum1, sum2 := sums1[i], sums2[i]
		if sum1 < sum2 {
			res += sum2
		} else {
			res += sum1
		}
	}
	return res
}

func GetCommon(nums1 []int, nums2 []int) map[int]bool {
	d := map[int]bool{}
	for _, num := range nums1 {
		d[num] = true
	}
	common := map[int]bool{}
	for _, num := range nums2 {
		if _, ok := d[num]; ok {
			common[num] = true
		}
	}
	return common
}

func GetMiddleSum(nums []int, common map[int]bool) []int {
	sum := 0
	path := []int{}
	for _, num := range nums {
		if _, ok := common[num]; ok {
			path = append(path, sum)
			sum = 0
		} else {
			sum += num
		}
	}
	path = append(path, sum)
	return path
}

func main() {
	nums1 := []int{2, 4, 5, 8, 10}
	nums2 := []int{4, 6, 8, 9}
	res := maxSum(nums1, nums2)
	fmt.Println(res)
}
