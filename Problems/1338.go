package main

import (
	"fmt"
	"sort"
)

func minSetSize(arr []int) int {
	count := make(map[int]int)
	for _, num := range arr {
		count[num] += 1
	}
	values := []int{}
	for _, value := range count {
		values = append(values, value)
	}
	sort.Slice(values, func(i, j int) bool {
		if values[i] > values[j] {
			return true
		}
		return false
	})
	target := (len(arr) + 1) / 2
	sum := 0
	res := 0
	for _, value := range values {
		sum += value
		res += 1
		if sum >= target {
			return res
		}
	}
	return -1
}

func main() {
	arr := []int{3, 3, 3, 3, 5, 5, 5, 2, 2, 7}
	res := minSetSize(arr)
	fmt.Println(res)
}
