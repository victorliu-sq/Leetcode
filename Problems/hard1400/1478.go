package main

import (
	"fmt"
	"math"
	"sort"
)

func minDistance(houses []int, k int) int {
	// Get cost[i][j] -> min cost by stubbing a mailbox between house i and house j
	sort.Slice(houses, func(i, j int) bool {
		if houses[i] <= houses[j] {
			return true
		}
		return false
	})
	cost := GetCost(houses)
	fmt.Println(cost)
	// try to divide houses into k parts with min cost
	dp := make(map[int]int)
	return dfs1478(0, 0, len(houses), k, cost, dp)
}

func dfs1478(curIdx, curGroup, n, k int, cost [][]int, dp map[int]int) int {
	// Base case
	if curIdx == n && curGroup == k {
		return 0
	}
	if curIdx == n || curGroup > k || curGroup+(n-curIdx) < k {
		return math.MaxInt32
	}
	// check dp
	key := GetKey1478(curIdx, curGroup)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// Next Recursion
	i := curIdx
	res := math.MaxInt32
	for j := i; j < n; j++ {
		curCost := cost[i][j]
		temp := curCost + dfs1478(j+1, curGroup+1, n, k, cost, dp)
		if temp < res {
			res = temp
		}
	}
	dp[key] = res
	return res
}

func GetKey1478(curIdx, curGroup int) int {
	// curIdx -> n -> 100 -> 7
	// curGroup -> k -> 100 -> 7
	key := 0
	key += curIdx
	key += curGroup << 7
	return key
}

func GetCost(houses []int) [][]int {
	n := len(houses)
	cost := make([][]int, n)
	for i := range cost {
		cost[i] = make([]int, n)
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			midPos := float64(houses[(i+j)/2])
			temp := 0
			for k := i; k <= j; k++ {
				temp += int(math.Abs(float64(houses[k]) - midPos))
			}
			cost[i][j] = temp
		}
	}
	return cost
}

func main() {
	houses := []int{2, 3, 5, 12, 18}
	k := 2
	res := minDistance(houses, k)
	fmt.Println(res)
}
