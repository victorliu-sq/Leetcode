package main

import "fmt"

func makeConnected(n int, connections [][]int) int {
	if len(connections) < n-1 {
		return -1
	}
	parent := make([]int, n)
	for i := 0; i < n; i++ {
		parent[i] = i
	}
	// (1) Get total num of groups
	groupNum := n
	for _, connection := range connections {
		a, b := connection[0], connection[1]
		pa, pb := find(parent, a), find(parent, b)
		if pa != pb {
			parent[pa] = pb
			// fmt.Println("Hello")
			groupNum -= 1
		}
	}
	// (2) Edges that we need to change == num of remaining groups - 1
	return groupNum - 1
}

func find(parent []int, node int) int {
	for parent[node] != node {
		parent[node] = find(parent, parent[node])
		node = parent[node]
	}
	return parent[node]
}

func main() {
	connection := [][]int{{0, 1}, {0, 2}, {1, 2}}
	res := makeConnected(4, connection)
	fmt.Println(res)
}
