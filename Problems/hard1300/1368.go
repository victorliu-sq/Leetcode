package main

import (
	"container/heap"
	"fmt"
)

func minCost(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dirs := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}

	h := &MinHeap1368{}
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}

	heap.Push(h, Node{0, 0, grid[0][0], 0})
	for h.Len() > 0 {
		// fmt.Println(h)
		// pop out node with min cost and set that position as visited
		node := heap.Pop(h).(Node)
		visited[node.X][node.Y] = true
		// fmt.Println(node)
		// if node is end, return cost
		if node.X == m-1 && node.Y == n-1 {
			return node.Cost
		}
		// iterate each dir
		for i := 1; i <= 4; i++ {
			newX := node.X + dirs[i-1][0]
			newY := node.Y + dirs[i-1][1]
			newCost := node.Cost
			// check whether Dir of node == grid[x][y], if not ,cost += 1
			if i != node.Dir {
				newCost++
			}
			if 0 <= newX && newX < m && 0 <= newY && newY < n {
				if !visited[newX][newY] {
					heap.Push(h, Node{newX, newY, grid[newX][newY], newCost})
				}
			}
		}
	}
	return 0
}

type Node struct {
	X    int
	Y    int
	Dir  int
	Cost int
}

type MinHeap1368 []Node

func (h *MinHeap1368) Len() int {
	return len(*h)
}

func (h *MinHeap1368) Less(i, j int) bool {
	return (*h)[i].Cost <= (*h)[j].Cost
}

func (h *MinHeap1368) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

// we must use interface{} instead of int
func (h *MinHeap1368) Push(x interface{}) {
	*h = append(*h, x.(Node))
}

func (h *MinHeap1368) Pop() interface{} {
	x := (*h)[len(*h)-1]
	(*h) = (*h)[:len(*h)-1]
	return x
}

func main() {
	// grid := [][]int{{1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}}
	grid := [][]int{{1, 1, 3}, {3, 2, 2}, {1, 1, 4}}
	res := minCost(grid)
	fmt.Println(res)
}
