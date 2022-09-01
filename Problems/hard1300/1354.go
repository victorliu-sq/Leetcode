package main

import (
	"container/heap"
	"fmt"
)

/*
Each time erase the max num and push a new candidate into heap

if maxNum == 1, all nums in array are 1 and we can return true
if next time, candidate must be 0 or negative, return false

trick: when we get next candidate, we get (curMax-1)%subTotal + 1 rather than curMax % subTotal
eg [9, 3] --> [3, 3] not [3, 0]
*/

func isPossible(target []int) bool {
	// [2] --> only 1 element and no chance to get [1]
	if len(target) == 1 && target[0] != 1 {
		return false
	}
	maxHeap := &MaxHeap{}
	for _, num := range target {
		heap.Push(maxHeap, num)
	}
	fmt.Println(*maxHeap)
	total := maxHeap.Sum()
	for {
		curMax := heap.Pop(maxHeap).(int)
		if curMax == 1 {
			return true
		}
		subTotal := total - curMax
		// [2, 1, 2] / [2, 1, 1] cannot be simplified
		if curMax <= subTotal {
			return false
		}
		// (1) [1000000, 1] -> timeout
		// candidate = curMax - subTotal => curMax % subTotal => (curMax - 1) % subTotal + 1
		candidate := (curMax-1)%subTotal + 1
		total = total - curMax + candidate
		heap.Push(maxHeap, candidate)
		fmt.Println(*maxHeap)
	}
}

type MaxHeap []int

func (h *MaxHeap) Len() int {
	return len(*h)
}

func (h *MaxHeap) Less(i, j int) bool {
	return (*h)[i] >= (*h)[j]
}

func (h *MaxHeap) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}

func (h *MaxHeap) Pop() interface{} {
	result := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]

	return result
}

func (h *MaxHeap) Sum() int {
	sum := 0
	for _, num := range *h {
		sum += num
	}
	return sum
}

func main() {
	target := []int{9, 3, 5}
	res := isPossible(target)
	fmt.Println(res)
}
