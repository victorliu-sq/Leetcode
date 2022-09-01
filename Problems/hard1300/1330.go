package main

import (
	"fmt"
	"math"
)

/*
abs(a - b) = max(a, b) - min(a, b)
(1) reverse subarray [0 ... i] i + 1 or i [i + 1  ... n - 1]
	change = abs(nums[i + 1] - nums[0]) - abs(nums[i + 1] - nums[i])
	or
	change = abs(nums]n  - 1 - nums[i]) - abs(nums[i + 1] - nums[i])

(2) reverse subarray nums[i] {nums[i + 1] ... nums[j]} nums[j + 1]
	original: max1 - min1 + max2 - min2
	after changing: 1 of max will become negative -> - max_, 1 of min will become positive -> +min_
	change: 2 * min_ - 2 * max_
	to maximize change, we need to get max min_ and min max_
*/

func maxValueAfterReverse(nums []int) int {
	changeSide := GetMaxChangeSide(nums)
	// fmt.Println(changeSide)
	changeMiddle := GetMaxChangeMiddle(nums)
	// fmt.Println(changeMiddle)
	total := GetTotal(nums)
	change := changeSide
	if change < changeMiddle {
		change = changeMiddle
	}
	if change < 0 {
		change = 0
	}
	return total + change
}

func GetMaxChangeSide(nums []int) int {
	n := len(nums)
	changeSide := -10000000
	for i := 0; i < n-1; i++ {
		change1 := GetAbs(nums[0], nums[i+1]) - GetAbs(nums[i], nums[i+1])
		if change1 > changeSide {
			changeSide = change1
		}
		change2 := GetAbs(nums[i], nums[n-1]) - GetAbs(nums[i], nums[i+1])
		if change2 > changeSide {
			changeSide = change2
		}
	}
	return changeSide
}

func GetMaxChangeMiddle(nums []int) int {
	n := len(nums)
	maxMin := -10000000
	minMax := 10000000
	for i := 0; i < n-1; i++ {
		a, b := nums[i], nums[i+1]
		var max, min int
		if a > b {
			max = a
			min = b
		} else {
			max = b
			min = a
		}
		if max < minMax {
			minMax = max
		}
		if min > maxMin {
			maxMin = min
		}
	}
	return 2 * (maxMin - minMax)
}

func GetTotal(nums []int) int {
	n := len(nums)
	sum := 0
	for i := 0; i < n-1; i++ {
		sum += GetAbs(nums[i], nums[i+1])
	}
	return sum
}

func GetAbs(n1 int, n2 int) int {
	return int(math.Abs(float64(n1) - float64(n2)))
}

func main() {
	nums := []int{2, 3, 1, 5, 4}
	res := maxValueAfterReverse(nums)
	fmt.Println(res)
}
