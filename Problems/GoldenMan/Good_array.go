package main

import (
	"fmt"
	"sort"
)

func GoodArray(N int, queries [][]int) []int {
	// (1) get GoodArray of N
	goodArray := GetGoodArray(N)
	// fmt.Println(goodArray)
	// (2) for each query, compute out the mod result
	res := []int{}
	for _, query := range queries {
		l, r, mod := query[0]-1, query[1]-1, query[2]
		product := 1
		if l < 0 {
			l = 0
		}
		if r >= len(goodArray) {
			r = len(goodArray) - 1
		}
		for i := l; i <= r; i++ {
			product *= goodArray[i]
		}
		res = append(res, product%mod)
	}
	return res
}

func GetGoodArray(N int) []int {
	res := []int{}
	for N > 0 {
		cur := GetMaxTwoPower(N)
		res = append(res, cur)
		N -= cur
	}
	sort.Ints(res)
	fmt.Println(res)
	return res
}

func GetMaxTwoPower(N int) int {
	cur := 1
	for cur < N {
		cur = cur * 2
	}
	if cur == N {
		return cur
	}
	return cur / 2
}

func main() {
	N := 26
	queries := [][]int{{1, 2, 1009}, {3, 3, 5}}
	res := GoodArray(N, queries)
	fmt.Println(res)
}
