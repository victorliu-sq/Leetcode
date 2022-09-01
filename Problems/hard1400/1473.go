package main

import (
	"fmt"
	"math"
)

/*
DFS: dp + dfs

Time complexity: curIdx * curGroup * prevColor * iteration = O(m * target * n * n)
Space complexity: curIdx * curGroup * prevColor
*/

func minCost(houses []int, cost [][]int, m int, n int, target int) int {
	colors := []int{}
	dp := map[int]int{}
	res := dfs1473(0, m, 0, target, -1, n, colors, houses, cost, dp)
	if res == math.MaxInt32 {
		return -1
	}
	return res
}

func dfs1473(curIdx, m, curGroup, target, prevColor, n int, colors, houses []int, cost [][]int, dp map[int]int) int {
	// 1. base case
	// fmt.Println(curIdx, prevColor, curGroup, colors)
	// (1) if curGroup exceeds taregt  or not enough house to assign groups
	if curGroup > target || (curGroup+m-curIdx) < target {
		return math.MaxInt32
	}
	// (2) if all houses have been assigned and curGroup == target
	if curIdx == m {
		if curGroup == target {
			return 0
		} else {
			return math.MaxInt32
		}
	}
	// fmt.Println(curIdx, prevColor, curGroup, colors)
	// 2. check dp
	key := GetKey(curIdx, curGroup, prevColor)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// 3. next recursion
	// (1) cur house is painted
	if houses[curIdx] != 0 {
		newGroup := curGroup
		if prevColor != houses[curIdx] {
			newGroup += 1
		}
		newColors := colors
		newColors = append(newColors, houses[curIdx])
		// fmt.Println(newColors, newGroup)
		dp[key] = dfs1473(curIdx+1, m, newGroup, target, houses[curIdx], n, newColors, houses, cost, dp)
		return dp[key]
	}

	res := math.MaxInt32
	// (2) cur house is not painted
	for color := 1; color <= n; color++ {
		newGroup := curGroup
		if color != prevColor {
			newGroup++
		}
		newColors := colors
		newColors = append(newColors, color)
		temp := cost[curIdx][color-1] + dfs1473(curIdx+1, m, newGroup, target, color, n, newColors, houses, cost, dp)
		// fmt.Println(curIdx, newColors, newGroup)
		res = min1473(res, temp)
	}
	// 4. update dp
	dp[key] = res
	return dp[key]
}

func GetKey(curIdx, curGroup, prevColor int) int {
	// curIdx -> m : 100 -> 7
	// prevColor -> n : 20 -> 5
	// curGroup -> m: 100 -> 7
	// [1, 2, 2], [1, 1, 2]
	key := 0
	key += curIdx
	key += curGroup << 7
	key += prevColor << 12
	// fmt.Println(curIdx, prevColor, curGroup, key)
	return key
}

func min1473(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	house := []int{0, 0, 0, 0, 0}
	cost := [][]int{{1, 10}, {10, 1}, {10, 1}, {1, 10}, {5, 1}}
	m := 5
	n := 2
	target := 3
	res := minCost(house, cost, m, n, target)
	fmt.Println(res)
}
