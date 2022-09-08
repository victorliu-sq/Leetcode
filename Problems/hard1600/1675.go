package main

import (
	"container/heap"
	"fmt"
	"math"
)

/*
	1. 	Change start by all odd num *= 2,
		since all possible arrays are still reachable with new start,
		the result will not be affected

	2.	in the new arr [biggest maxNum, ... biggest minNum]
		we can only get a possibly smaller difference by decreasing maxNum / increasing minNum
		In this way, pop out biggest maxNum and divide it by 2 until biggest maxNum is odd
		(odd means that we cannot choose to decrease biggest maxNum anymore)

*/

func minimumDeviation(nums []int) int {
	h := &maxHeap{}
	// 1. Change start
	// Get minNum and add all nums into maxHeap
	minNum := math.MaxInt32
	for _, num := range nums {
		if num%2 == 1 {
			num *= 2
		}
		heap.Push(h, num)
		minNum = min1675(minNum, num)
		fmt.Println(h)
	}
	// 2. Iterate maxNum one by one, update difference and minNum
	res := math.MaxInt32
	for h.Len() > 0 {
		maxNum := heap.Pop(h).(int)
		fmt.Println(maxNum, minNum)
		res = min1675(res, maxNum-minNum)
		if maxNum%2 == 1 {
			break
		}
		newNum := maxNum / 2
		minNum = min1675(minNum, newNum)
		heap.Push(h, newNum)
		fmt.Println(h)
	}
	return res
}

func min1675(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

// maxHeap
type maxHeap []int

func (h maxHeap) Less(i, j int) bool {
	return h[i] >= h[j]
}

func (h maxHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h maxHeap) Len() int {
	return len(h)
}

func (h *maxHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}

func (h *maxHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	(*h) = (*h)[:len(*h)-1]
	return x
}

func main() {
	nums := []int{4, 1, 5, 20, 3}
	res := minimumDeviation(nums)
	fmt.Println(res)
}
