package main

import "fmt"

// 45 jump game2
// [left ... right] => [right + 1 ... nextRight]
// nextRight comes [left ... right]
func jump(nums []int) int {
	n := len(nums)
	if n == 1 {
		return 0
	}
	res := 1
	left, right := 0, nums[0]
	for right < n-1 {
		var nextRight int
		for i := left; i <= right; i++ {
			if i+nums[i] > nextRight {
				nextRight = i + nums[i]
			}
		}
		fmt.Println(right, nextRight)
		if nextRight == right {
			return -1
		}
		res++
		left, right = right+1, nextRight
	}
	return res
}

// 1326
// convert ranges into jump games
func minTaps(n int, ranges []int) int {
	arr := GetJump2Arr(n, ranges)
	// fmt.Println(arr)
	res := jump(arr)
	return res
}

func GetJump2Arr(n int, ranges []int) []int {
	arr := make([]int, n+1)
	for pos, r := range ranges {
		// Get left, right boundary and range of these two boundaries
		var left, right int
		if pos-r > 0 {
			left = pos - r
		} else {
			left = 0
		}
		if pos+r <= n {
			right = pos + r
		} else {
			right = n
		}
		arr[left] = right - left
	}
	return arr
}

func main() {
	// 45
	// nums := []int{2, 3, 1, 1, 4}
	// res1 := jump(nums)
	// fmt.Println(res1)
	// 1326
	ranegs := []int{0, 0, 0, 0}
	n := 3
	res2 := minTaps(n, ranegs)
	fmt.Println(res2)
}
