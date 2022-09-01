package main

import "fmt"

func cherryPickup(grid [][]int) int {
	dp := map[int]int{}
	return dfs1463(0, 0, len(grid[0])-1, grid, dp)
}

func dfs1463(curRow, col1, col2 int, grid [][]int, dp map[int]int) int {
	// base case
	m, n := len(grid), len(grid[0])
	if curRow == m {
		return 0
	}
	// check dp
	key := GetKey(curRow, col1, col2)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// next recursion
	cols1 := GetNextCols(col1, n)
	cols2 := GetNextCols(col2, n)
	sumCurRow := 0
	if col1 == col2 {
		sumCurRow = grid[curRow][col1]
	} else {
		sumCurRow = grid[curRow][col1] + grid[curRow][col2]
	}
	temp := 0
	for _, newCol1 := range cols1 {
		for _, newCol2 := range cols2 {
			temp = max1463(temp, dfs1463(curRow+1, newCol1, newCol2, grid, dp))
		}
	}
	// update dp
	dp[key] = sumCurRow + temp
	return sumCurRow + temp
}

func GetKey(curRow, col1, col2 int) int {
	// 1 <= rows , cols <= 70
	key := 0
	key += curRow
	key += col1 << 7
	key += col2 << 14
	return key
}

func GetNextCols(col, n int) []int {
	cols := []int{}
	cols = append(cols, col)
	if col > 0 {
		cols = append(cols, col-1)
	}
	if col < n-1 {
		cols = append(cols, col+1)
	}
	return cols
}
func max1463(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	grid := [][]int{{3, 1, 1}, {2, 5, 1}, {1, 5, 5}, {2, 1, 1}}
	res := cherryPickup(grid)
	fmt.Println(res)
}
