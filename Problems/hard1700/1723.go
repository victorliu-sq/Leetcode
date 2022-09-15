package main

import (
	"fmt"
	"math"
)

/*
473: https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
698: https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
*/

func minimumTimeRequired(jobs []int, k int) int {
	res := math.MaxInt32
	workingTime := make([]int, k)
	dfs(0, jobs, workingTime, &res)
	return res
}

func dfs(curIndex int, jobs, workingTime []int, res *int) {
	n := len(jobs)
	k := len(workingTime)
	// base case
	if curIndex == n {
		maxWorkingTime := 0
		for _, Time := range workingTime {
			maxWorkingTime = max1723(maxWorkingTime, Time)
		}
		*res = min1723(*res, maxWorkingTime)
		return
	}
	// next recursion
	m := map[int]int{}
	fmt.Println(curIndex, workingTime)
	// choose one worker to assign current job
	for i := 0; i < k; i++ {
		// if current job has been assigned to a worker with recurring workingTime
		// ignore this worker
		if _, ok := m[workingTime[i]]; ok {
			continue
		}
		m[workingTime[i]] = 1
		if workingTime[i]+jobs[curIndex] >= *res {
			break
		}
		workingTime[i] += jobs[curIndex]
		dfs(curIndex+1, jobs, workingTime, res)
		workingTime[i] -= jobs[curIndex]
	}
}

func max1723(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func min1723(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	jobs := []int{1, 2, 4, 7, 8}
	k := 2
	res := minimumTimeRequired(jobs, k)
	fmt.Println(res)
}
