package main

import (
	"fmt"
	"sort"
)

func FootballScores(A []int, B []int) []int {
	sort.Ints(A)
	sort.Ints(B)
	n := len(A)
	res := []int{}
	for _, score := range B {
		left, right := 0, n-1
		for left < right {
			mid := left + (right-left)/2 + 1
			if A[mid] <= score {
				left = mid
			} else {
				right = mid - 1
			}
		}
		res = append(res, A[left])
	}
	return res
}

func main() {
	A := []int{1, 2, 3}
	B := []int{2, 4}
	res := FootballScores(A, B)
	fmt.Println(res)
}
