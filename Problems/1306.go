package main

import (
	"container/list"
	"fmt"
)

func canReach(arr []int, start int) bool {
	length := len(arr)
	queue := list.New()
	visited := make(map[int]bool)
	queue.PushBack(start)
	visited[start] = true
	for queue.Len() != 0 {
		size := queue.Len()
		for i := 0; i < size; i++ {
			// curIdx := Pop(queue)
			curIdxP := queue.Front()
			queue.Remove(curIdxP)
			curIdx := curIdxP.Value.(int)
			// fmt.Println(curIdx)
			if arr[curIdx] == 0 {
				return true
			}
			// queue.
			next1 := curIdx + arr[curIdx]
			next2 := curIdx - arr[curIdx]
			if _, ok := visited[next1]; !ok && next1 < length {
				queue.PushBack(next1)
				visited[next1] = true
			}
			if _, ok := visited[next2]; !ok && next2 >= 0 {
				queue.PushBack(next2)
				visited[next2] = true
			}
		}

	}
	return false
}

func Pop(queue *list.List) interface{} {
	curIdxP := queue.Front()
	queue.Remove(curIdxP)
	return curIdxP
}

func main() {
	arr := []int{4, 2, 3, 0, 3, 1, 2}
	start := 5
	res := canReach(arr, start)
	fmt.Println(res)
}
