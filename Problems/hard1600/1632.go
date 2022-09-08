package main

import (
	"fmt"
	"sort"
)

/*
1. 	iterate each num from small to big, union all rows and cols together to get smallest rank

2.	same num may exist in different groups, for each group, there is one different smallest rank

*/

/*
Trick
How to use union find to connect row i and col j
j = m + j, connect i with (m + j)
*/

func matrixRankTransform(matrix [][]int) [][]int {
	// return [][]int{}
	// Get all Coordinates of 1 num
	coords := GetCoordinates(matrix)
	fmt.Println(coords)
	keys := GetSortedKeys(coords)
	fmt.Println(keys)
	m, n := len(matrix), len(matrix[0])
	// Initialize ranks of all rows and cols to 0
	maxRanks := make([]int, m+n)
	// Get a result matrix to store ranks
	res := make([][]int, m)
	for i := range res {
		res[i] = make([]int, n)
	}
	// iterate num from small to big
	for _, key := range keys {
		fmt.Printf("current key is %v\n", key)
		// Get max rank of all rows and cols of this num
		parent := GetParent(m, n)
		for _, coord := range coords[key] {
			x, y := coord[0], coord[1]+m
			// fmt.Println(x, y)
			rx, ry := find(parent, x), find(parent, y)
			parent[ry] = rx
			maxRanks[rx] = max1632(maxRanks[rx], max1632(maxRanks[rx], maxRanks[ry]))
		}
		// update rank of all coordinates of this num
		tempMaxRanks := CopyMaxRanks(maxRanks)
		for _, coord := range coords[key] {
			x, y := coord[0], coord[1]+m
			rx, ry := find(parent, x), find(parent, y)
			maxRank := max1632(tempMaxRanks[rx], tempMaxRanks[ry]) + 1
			fmt.Println(rx, ry, "maxRank:", maxRank)
			res[x][y-m], maxRanks[x], maxRanks[y] = maxRank, maxRank, maxRank
		}
		// fmt.Println("maxRank is:", maxRank)
	}
	return res
}

func find(parent []int, i int) int {
	for parent[i] != i {
		parent[i] = find(parent, parent[i])
		i = parent[i]
	}
	return parent[i]
}

func GetCoordinates(matrix [][]int) map[int][][]int {
	coords := map[int][][]int{}
	for i := range matrix {
		for j := range matrix[i] {
			num := matrix[i][j]
			if _, ok := coords[num]; !ok {
				coords[num] = [][]int{}
			}
			coords[num] = append(coords[num], []int{i, j})
		}
	}
	return coords
}

func GetSortedKeys(coords map[int][][]int) []int {
	keys := []int{}
	for key, _ := range coords {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	return keys
}

func max1632(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func GetParent(m, n int) []int {
	parent := make([]int, m+n)
	for i := range parent {
		parent[i] = i
	}
	return parent
}

func CopyMaxRanks(maxRanks []int) []int {
	temp := []int{}
	for _, rank := range maxRanks {
		temp = append(temp, rank)
	}
	return temp
}

func main() {
	// matrix := [][]int{{1, 2}, {3, 4}}
	// matrix := [][]int{{20, -21, 14}, {-19, 4, 19}, {22, -47, 24}, {-19, 4, 19}}

	// The reason why we need union find
	// num 47 rank comes from (1) row 1 & col 3 (2) row 4 col 0
	// same num can comes from different rank
	matrix := [][]int{{-37, -26, -47, -40, -13}, {22, -11, -44, 47, -6}, {-35, 8, -45, 34, -31}, {-16, 23, -6, -43, -20}, {47, 38, -27, -8, 43}}
	res := matrixRankTransform(matrix)
	fmt.Println(res)
}
