package main

import (
	"fmt"
	"math"
)

/*
Brute force
*/

func minTrioDegree(n int, edges [][]int) int {
	graph := make([][]bool, n+1)
	for i := range graph {
		graph[i] = make([]bool, n+1)
	}
	// Get degree and graph
	degree := make([]int, n+1)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u][v] = true
		graph[v][u] = true
		degree[u] += 1
		degree[v] += 1
	}
	// fmt.Println(graph)
	// Iterate all pairs of points
	res := math.MaxInt32
	for i := 1; i <= n; i++ {
		for j := i + 1; j < n; j++ {
			// faster
			if !graph[i][j] {
				continue
			}
			for k := j + 1; k < n; k++ {
				if graph[i][j] && graph[j][k] && graph[i][k] {
					temp := degree[i] + degree[j] + degree[k] - 6
					fmt.Println(temp)
					res = min1761(res, temp)
				}
			}
		}
	}
	if res == math.MaxInt32 {
		res = -1
	}
	return res
}

func min1761(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	// n := 6
	// edges := [][]int{{1, 2}, {1, 3}, {3, 2}, {4, 1}, {5, 2}, {3, 6}}
	n := 4
	edges := [][]int{{1, 2}, {4, 1}, {4, 2}}
	res := minTrioDegree(n, edges)
	fmt.Println(res)
}
