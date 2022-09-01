package main

import "fmt"

/*
DFS:
(1) Get next rows
(2) for each row, choose one row that does not collide with last row
*/

func numOfWays(n int) int {
	rows := GetRows()
	fmt.Println(rows)
	dp := map[int]int{}
	res := dfs1411(0, n, 0, rows, dp)
	return res
}

func dfs1411(curIdx, n, lastRow int, rows []int, dp map[int]int) int {
	if curIdx == n {
		// fmt.Println(lastRow)
		return 1
	}
	key := curIdx<<10 + lastRow
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	res := 0
	for _, row := range rows {
		if CheckCollions(row, lastRow) {
			continue
		}
		res += dfs1411(curIdx+1, n, row, rows, dp)
	}
	dp[key] = res
	return res
}

func CheckCollions(row1, row2 int) bool {
	for row1 > 0 {
		color1, color2 := row1%10, row2%10
		if color1 == color2 {
			return true
		}
		row1 /= 10
		row2 /= 10
	}
	return false
}

func GetRows() []int {
	rows := []int{}
	dfs1411Rows(0, 0, -1, &rows)
	return rows
}

func dfs1411Rows(curIdx int, curRow, lastColor int, rows *[]int) {
	if curIdx == 3 {
		// fmt.Println(curRow)
		*rows = append(*rows, curRow)
		return
	}
	for color := 1; color <= 3; color++ {
		if color == lastColor {
			continue
		}
		dfs1411Rows(curIdx+1, curRow*10+color, color, rows)
	}
}

func main() {
	n := 10
	res := numOfWays(n)
	fmt.Println(res)
}
