package main

import (
	"fmt"
	"sort"
)

func distanceLimitedPathsExist(n int, edgeList [][]int, queries [][]int) []bool {
	m := len(queries)
	// sort edges by weight from small to big
	// edge -> 0 u, 1 v, 2 w
	sort.Slice(edgeList, func(i, j int) bool {
		if edgeList[i][2] <= edgeList[j][2] {
			return true
		}
		return false
	})
	// fmt.Println(edgeList)
	// sort queries(add idx) from small to big
	// query -> 0 p, 1 q, 2 limit, 3 idx
	for idx, _ := range queries {
		queries[idx] = append(queries[idx], idx)
	}
	sort.Slice(queries, func(i, j int) bool {
		if queries[i][2] <= queries[j][2] {
			return true
		}
		return false
	})
	// fmt.Println(queries)
	// for each query, union all edges of smaller weight from left to right
	// after union, check whether p and q are connected together
	edgeIdx := 0
	edgeNum := len(edgeList)
	parent := GetParent(n)
	res := make([]bool, m)
	for _, query := range queries {
		// while we can union
		for edgeIdx < edgeNum && edgeList[edgeIdx][2] < query[2] {
			u, v := edgeList[edgeIdx][0], edgeList[edgeIdx][1]
			Connect(parent, u, v)
			edgeIdx++
		}
		// fmt.Println(parent)
		p, q, idx := query[0], query[1], query[3]
		res[idx] = Find(parent, p) == Find(parent, q)
		// fmt.Println(res[idx], idx)
	}
	return res
}

func GetParent(n int) []int {
	parent := []int{}
	for i := 0; i < n; i++ {
		parent = append(parent, i)
	}
	return parent
}

func Find(parent []int, i int) int {
	for parent[i] != i {
		parent[i] = Find(parent, parent[i])
		i = parent[i]
	}
	return parent[i]
}

func Connect(parent []int, i, j int) {
	r1, r2 := Find(parent, i), Find(parent, j)
	if r1 != r2 {
		parent[r1] = r2
	}
}

func main() {
	n := 5
	edgeList := [][]int{{0, 1, 10}, {1, 2, 5}, {2, 3, 9}, {3, 4, 13}}
	queries := [][]int{{0, 4, 14}, {1, 4, 13}}
	res := distanceLimitedPathsExist(n, edgeList, queries)
	fmt.Println(res)
}
