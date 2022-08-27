package main

import "fmt"

func numTeams(rating []int) int {
	res := 0
	for mid := 0; mid < len(rating); mid++ {
		smallerL := GetSmallerLeft(rating, mid)
		smallerR := GetSmallerRight(rating, mid)
		biggerL := GetBiggerLeft(rating, mid)
		biggerR := GetBiggerRight(rating, mid)
		res += smallerL*biggerR + biggerL*smallerR
	}
	return res
}

func GetSmallerLeft(rating []int, mid int) int {
	smaller := 0
	for i := 0; i < mid; i++ {
		if rating[i] < rating[mid] {
			smaller++
		}
	}
	return smaller
}

func GetSmallerRight(rating []int, mid int) int {
	smaller := 0
	n := len(rating)
	for i := mid + 1; i < n; i++ {
		if rating[i] < rating[mid] {
			smaller++
		}
	}
	return smaller
}

func GetBiggerLeft(rating []int, mid int) int {
	bigger := 0
	for i := 0; i < mid; i++ {
		if rating[i] > rating[mid] {
			bigger++
		}
	}
	return bigger
}

func GetBiggerRight(rating []int, mid int) int {
	bigger := 0
	n := len(rating)
	for i := mid + 1; i < n; i++ {
		if rating[i] > rating[mid] {
			bigger++
		}
	}
	return bigger
}

func main() {
	ex := []int{2, 5, 3, 4, 1}
	res := numTeams(ex)
	fmt.Println(res)
}
