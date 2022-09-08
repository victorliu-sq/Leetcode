package main

import (
	"fmt"
)

func createSortedArray(instructions []int) int {
	costs := 0
	arr := []int{}
	for _, target := range instructions {
		smaller := FindLastSmaller(arr, target)
		bigger := FindFirstBigger(arr, target)
		fmt.Println("Insert:", target, "smaller:", smaller, "bigger:", bigger)
		cost := min1649(smaller, bigger)
		costs += cost
		// arr = append(arr, target)
		// sort.Ints(arr)
		s1, s2 := arr[:smaller], arr[smaller:]
		newArr := make([]int, len(arr)+1)
		// fmt.Println(s1, s2)

		copy(newArr, s1)
		// fmt.Println(newArr)

		newArr[smaller] = target
		copy(newArr[len(s1)+1:], s2)
		// fmt.Println(newArr)

		arr = newArr
		// fmt.Println(arr)
	}
	return costs
}

func FindLastSmaller(arr []int, target int) int {
	l, r := 0, len(arr)-1
	for l < r {
		// fmt.Println(l, r)
		m := l + (r-l)/2 + 1
		if arr[m] >= target {
			r = m - 1
		} else {
			l = m
		}
	}
	if len(arr) == 0 || arr[l] >= target {
		return 0
	}
	return l - 0 + 1
}

func FindFirstBigger(arr []int, target int) int {
	l, r := 0, len(arr)-1
	for l < r {
		// fmt.Println(l, r)
		m := l + (r-l)/2
		if arr[m] <= target {
			l = m + 1
		} else {
			r = m
		}
	}
	if len(arr) == 0 || arr[r] <= target {
		return 0
	}
	return len(arr) - 1 - r + 1
}

func min1649(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	instructions := []int{1, 5, 6, 2}
	res := createSortedArray(instructions)
	fmt.Println(res)
}
