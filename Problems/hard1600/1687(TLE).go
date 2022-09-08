package main

import (
	"fmt"
	"math"
)

/*
	This is a DFS + DP problem
	curIdx, curBoxes, curWright, lastPort

*/

func boxDelivering(boxes [][]int, portsCount int, maxBoxes int, maxWeight int) int {
	dp := map[int]int{}
	res := dfs1687(0, 0, 0, -1, boxes, maxBoxes, maxWeight, dp)
	return res
}

func dfs1687(curIdx, curBoxes, curWeight, lastPort int, boxes [][]int, maxBoxes int, maxWeight int, dp map[int]int) int {
	// 1. base case
	if curIdx == len(boxes) {
		// this is a new ship, it has returned --> no extra trips needed
		if lastPort == -1 {
			return 0
		}
		// this is an old ship
		return 1
	}
	// 2.check dp
	key := GetKey(curIdx, curBoxes, curWeight, lastPort)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// 3. next recursion
	// (1) add current box to an old ship
	res := math.MaxInt32
	curPort, curBoxWeight := boxes[curIdx][0], boxes[curIdx][1]
	if curBoxes+1 <= maxBoxes && curWeight+curBoxWeight <= maxWeight {
		trip := 0
		if lastPort != curPort {
			trip++
		}
		newIdx, newBoxes, newWeight, newPort := curIdx+1, curBoxes+1, curWeight+curBoxWeight, curPort
		temp := trip + dfs1687(newIdx, newBoxes, newWeight, newPort, boxes, maxBoxes, maxWeight, dp)
		res = min1687(res, temp)
	}
	// (2) add current box to a new ship
	newIdx, newBoxes, newWeight, newPort := curIdx+1, 1, curBoxWeight, curPort
	// return + go to new port
	temp := 1 + 1 + dfs1687(newIdx, newBoxes, newWeight, newPort, boxes, maxBoxes, maxWeight, dp)
	res = min1687(res, temp)
	// 4. update dp
	dp[key] = res
	return res
}

func min1687(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func GetKey(curIdx, curBoxes, curWeight, lastPort int) int {
	// all 10 **5 --> 2 ^ 20
	return curIdx + curBoxes*2 + curWeight*3 + lastPort*5
}

func main() {
	boxes := [][]int{{1, 1}, {2, 1}, {1, 1}}
	portsCount := 2
	maxBoxes := 3
	maxWeight := 3
	res := boxDelivering(boxes, portsCount, maxBoxes, maxWeight)
	fmt.Println(res)
}
