package main

import (
	"fmt"
	"math"
)

/*
	How to choose a subset: (1) choose n cities (2) add all edges in those cities
	since a subset should also be a tree: we need to check whether # edges == # vertexes - 1
*/

func countSubgraphsForEachDiameter(n int, edges [][]int) []int {
	// Get min dist between any 2 cities with Floyd-Warshall Algorithm
	dist := make([][]int, n)
	for i := range dist {
		dist[i] = make([]int, n)
		for j := range dist[i] {
			if i != j {
				dist[i][j] = math.MaxInt32
			}
		}
	}

	for _, edge := range edges {
		a, b := edge[0]-1, edge[1]-1
		dist[a][b] = 1
		dist[b][a] = 1
	}

	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				dist[i][j] = min1611(dist[i][j], dist[i][k]+dist[k][j])
				dist[j][i] = dist[i][j]
			}
		}
	}
	fmt.Println(dist)
	res := make([]int, n-1)
	// Get max distance of any subset
	for state := 1; state < (1<<(n+1) - 1); state++ {
		edgeNum := 0
		maxDist := 0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if state&(1<<i) == 0 || state&(1<<j) == 0 {
					continue
				}
				if dist[i][j] == 1 {
					edgeNum++
				}
				maxDist = max1611(maxDist, dist[i][j])
			}
		}
		if maxDist >= 1 && edgeNum == GetCityNum(state)-1 {
			// fmt.Printf("subset is %v and maxDist is %v\n", strconv.FormatInt(int64(state), 2), maxDist)
			res[maxDist-1]++
		}
	}
	return res
}

func GetCityNum(state int) int {
	cityNum := 0
	for state > 0 {
		if state%2 == 1 {
			cityNum++
		}
		state /= 2
	}
	return cityNum
}

func min1611(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func max1611(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	n := 4
	edges := [][]int{{1, 2}, {2, 3}, {2, 4}}
	res := countSubgraphsForEachDiameter(n, edges)
	fmt.Println(res)
}
