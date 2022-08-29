package main

import "fmt"

func findTheCity(n int, edges [][]int, distanceThreshold int) int {
	dist := make([][]int, n)
	// initialize dist with max dist
	for i := range dist {
		dist[i] = make([]int, n)
		for j := range dist[i] {
			dist[i][j] = 100000
		}
	}
	// set to 0
	for i := 0; i < n; i++ {
		dist[i][i] = 0
	}
	// set to weight
	for _, edge := range edges {
		src, dest, weight := edge[0], edge[1], edge[2]
		dist[src][dest] = weight
		dist[dest][src] = weight
	}
	reachable := make(map[int]int)
	fmt.Println(reachable)
	// update min dist from i to j
	// ** Notice that we need to loop k at first for Floyd Algorithm **
	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if dist[i][k]+dist[k][j] < dist[i][j] {
					dist[i][j] = dist[i][k] + dist[k][j]
				}
			}
		}
	}
	// Get reachable cities of each node
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if dist[i][j] <= distanceThreshold {
				reachable[i] += 1
			}
		}
	}
	fmt.Println(reachable)
	// find node with min count with max num
	res := -1
	minCount := n + 1
	for num := 0; num < n; num++ {
		if reachable[num] < minCount || (reachable[num] == minCount && num > res) {
			res = num
			minCount = reachable[num]
		}
	}
	return res
}

func main() {
	n := 4
	edges := [][]int{{0, 1, 3}, {1, 2, 1}, {1, 3, 4}, {2, 3, 1}}
	distanceThreshold := 4
	res := findTheCity(n, edges, distanceThreshold)
	fmt.Println(res)
}
