package main

import (
	"container/list"
	"fmt"
)

/*
DQ --> O(n)
dq --> 	(1) idx from small to big --> further will be popped out from front
		(2) sum of subsequence from big to small --> smaller will be popped out from back
*/

func constrainedSubsetSum(nums []int, k int) int {
	n := len(nums)
	dq := list.New()
	res := -1000000
	for i := 0; i < n; i++ {
		sum := 0
		// pop out further nodes from front by idx
		for dq.Len() > 0 {
			frontP := dq.Front()
			if i-frontP.Value.(Node1425).Idx > k {
				dq.Remove(frontP)
				continue
			}
			sum = max1425(sum, frontP.Value.(Node1425).Sum)
			break
		}
		newNode := Node1425{i, sum + nums[i]}
		// fmt.Println(newNode)
		if newNode.Sum > res {
			res = newNode.Sum
		}
		// pop out smalled nodes from back by value
		for dq.Len() > 0 {
			backP := dq.Back()
			if backP.Value.(Node1425).Sum < newNode.Sum {
				dq.Remove(backP)
			} else {
				break
			}
		}
		// push newNode into dq
		dq.PushBack(newNode)
		// pop out from back by sum
	}
	return res
}

type Node1425 struct {
	Idx int
	Sum int
}

func max1425(i, j int) int {
	if i > j {
		return i
	}
	return j
}

/*
DP --> O(n * k) --> O(n)
dp[i]: num[i] + max sequence afterwards

func constrainedSubsetSum(nums []int, k int) int {
	n := len(nums)
	dp := make([]int, n)
	for i := range dp {
		dp[i] = -10000
	}
	maxDp := -10000
	for i := n - 1; i >= 0; i-- {
		var temp int
		for j := 1; j <= k; j++ {
			if i+j <= n-1 && dp[i+j] > temp {
				temp = dp[i+j]
			}
			dp[i] = max1425(dp[i], nums[i]+temp)
		}
		maxDp = max1425(maxDp, dp[i])
	}
	res := -10000
	for i := range dp {
		res = max1425(res, dp[i])
	}
	return res
}
*/

func main() {
	k := 2
	// nums := []int{10, 2, -10, 5, 20}
	nums2 := []int{-8269, 3217, -4023, -4138, -683, 6455, -3621, 9242, 4015, -3790}
	res := constrainedSubsetSum(nums2, k)
	fmt.Println(res)
}
