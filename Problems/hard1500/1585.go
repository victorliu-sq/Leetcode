package main

import (
	"fmt"
	"sort"
)

/*
	if s = "1 2 3" and t = "3 2 1"
	for the first digit "3" , to move "3" of s to first position,
	there can not be any digits smaller than "2" in s on the left of "3"
	because we cannot move 1 / 2 to right of 3

	After moving 1 digit in s(if there are mutiple the same digit, move the leftmost one) to leftmost position, the sort remaining nums is relatively same as before
	--> we only need to delete cur digit to update index

	if s = "2, 3, 1" and t = "1, 2, 3"
	for the first digit 1, after moving 1 to leftmost position
	s -> "[1], 2, 3"
*/

func isTransformable(s string, t string) bool {
	// store digit and its indices in t
	indices := map[int][]int{}
	for idx, _ := range s {
		digit := int(s[idx] - '0')
		indices[digit] = append(indices[digit], idx)
	}
	keys := GetSortedKeys(indices)
	// fmt.Println(indices)
	// for each digit in t, check whether there exists a smaller num on the left
	for idx, _ := range t {
		digit := int(t[idx] - '0')
		// if no corresponding digit in s
		if _, ok := indices[digit]; !ok {
			return false
		}
		// we choose the leftmost digit in s
		indexCur := indices[digit][0]
		// iterate all smaller nums
		for _, key := range keys {
			if key >= digit {
				break
			}
			// check if key has a smaller index in s
			// fmt.Println(digit, key, indices)
			if len(indices[key]) > 0 {
				fmt.Println("curDigit:", string(t[idx]), "curIndex:", indexCur, "smallerDigit", key, "smallerIndex:", indices[key][0])
				if indices[key][0] < indexCur {
					return false
				}
			}
		}
		// if there is no smaller digit with a smaller index, we can move it to leftmost position by deleting it from map
		indices[digit] = indices[digit][1:]
		if len(indices[digit]) == 0 {
			delete(indices, digit)
		}
	}
	return true
}

func GetSortedKeys(indices map[int][]int) []int {
	keys := []int{}
	for key, indices := range indices {
		sort.Ints(indices)
		keys = append(keys, key)
	}
	sort.Ints(keys)
	return keys
}

func main() {
	s := "84532"
	t := "34852"
	res := isTransformable(s, t)
	fmt.Println(res)
}
