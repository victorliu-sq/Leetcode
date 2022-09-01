package main

import (
	"container/heap"
	"fmt"
)

/*
each node is col indices of all rows, next node come from any col + 1
*/

func kthSmallest(mat [][]int, k int) int {
	h := &MinHeap1439{}
	n := len(mat[0])
	// initialize heap
	total := 0
	cols := []int{}
	for i := range mat {
		total += mat[i][0]
		cols = append(cols, 0)
	}
	visited := map[string]bool{}
	// we do not want to add duplicate node into visited, so we set it as visited when pushing it in
	heap.Push(h, Node1439{total, cols})
	visited[GetKey(cols)] = true
	// each time pop out node with smallest value and push next element into heap
	i := 1
	for h.Len() > 0 {
		node := heap.Pop(h).(Node1439)
		fmt.Println(node.Sum)
		if i == k {
			return node.Sum
		}
		for row, col := range node.Cols {
			newCols := CopySlice(node.Cols)
			// fmt.Println(newCols)
			if col+1 <= n-1 {
				newCols[row] = col + 1
				newSum := node.Sum - mat[row][col] + mat[row][col+1]
				if _, ok := visited[GetKey(newCols)]; !ok {
					heap.Push(h, Node1439{newSum, newCols})
					visited[GetKey(newCols)] = true
				}
			}
		}
		fmt.Println(h)
		i++
	}
	// fmt.Println(h)
	return total
}

func GetKey(cols []int) string {
	key := ""
	for _, num := range cols {
		key += string(rune(num))
	}
	return key
}

func CopySlice(s []int) []int {
	newS := []int{}
	for _, num := range s {
		newS = append(newS, num)
	}
	return newS
}

type Node1439 struct {
	Sum  int
	Cols []int
}

// min Heap
type MinHeap1439 []Node1439

func (h *MinHeap1439) Len() int {
	return len(*h)
}

func (h *MinHeap1439) Less(i, j int) bool {
	return (*h)[i].Sum <= (*h)[j].Sum
}

func (h *MinHeap1439) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

// we must use interface{} instead of int
func (h *MinHeap1439) Push(x interface{}) {
	*h = append(*h, x.(Node1439))
}

func (h *MinHeap1439) Pop() interface{} {
	x := (*h)[len(*h)-1]
	(*h) = (*h)[:len(*h)-1]
	return x
}

func main() {
	mat := [][]int{{1, 3, 11}, {2, 4, 6}}
	k := 9
	res := kthSmallest(mat, k)
	fmt.Println(res)
}
