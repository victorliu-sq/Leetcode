package main

import (
	"fmt"
	"math"
)

func connectTwoGroups(cost [][]int) int {
	minCostNode2 := GetMinCostNode2(cost)
	fmt.Println(minCostNode2)
	res := dfs1595(0, 0, cost, minCostNode2, map[int]int{})
	return res
}

func dfs1595(curNode1, bitmaskNode2 int, cost [][]int, minCostNode2 map[int]int, dp map[int]int) int {
	n1, n2 := len(cost), len(cost[0])
	// base case
	if curNode1 == n1 {
		// if all node1 have been connected, connect all unconnected node2
		res := 0
		for node2 := 0; node2 < n2; node2++ {
			if bitmaskNode2&(1<<node2) == 0 {
				res += minCostNode2[node2]
			}
		}
		return res
	}
	// check dp
	key := GetKey(curNode1, bitmaskNode2)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// if not all node1 have been connected, choose 1 node2 to connect
	// if curNode1 < n1 {}
	res := math.MaxInt32
	for node2 := 0; node2 < n2; node2++ {
		newNode := curNode1 + 1
		newBitMaskNode2 := bitmaskNode2 | (1 << node2)
		res = min1595(res, cost[curNode1][node2]+dfs1595(newNode, newBitMaskNode2, cost, minCostNode2, dp))
	}
	dp[key] = res
	return res
}

func GetMinCostNode2(cost [][]int) map[int]int {
	n1, n2 := len(cost), len(cost[0])
	minCostNode2 := map[int]int{}
	for j := 0; j < n2; j++ {
		for i := 0; i < n1; i++ {
			if _, ok := minCostNode2[j]; !ok {
				minCostNode2[j] = cost[i][j]
			} else {
				minCostNode2[j] = min1595(minCostNode2[j], cost[i][j])
			}
		}
	}
	return minCostNode2
}

func GetKey(curIdx, bitmaskNode2 int) int {
	// curIdx -> 12 -> 4
	// bitmaskNode2 -> whatever
	return curIdx + bitmaskNode2<<4
}

func min1595(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	cost := [][]int{{15, 96}, {36, 2}}
	res := connectTwoGroups(cost)
	fmt.Println(res)
}
