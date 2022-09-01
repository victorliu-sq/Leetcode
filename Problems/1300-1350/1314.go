package main

import "fmt"

/*
Time Complexity: N ^ 2
preSum: sum[i][j] = preSum[i][j] + preSum[i - 2k][j] + preSum[i][j - 23] - preSum

*/

func matrixBlockSum(mat [][]int, k int) [][]int {
	m, n := len(mat), len(mat[0])
	preSum := GetNew2DArray(m+1, n+1)
	res := GetNew2DArray(m, n)
	// Get PreSum
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			preSum[i][j] = mat[i-1][j-1] + preSum[i-1][j] + preSum[i][j-1] - preSum[i-1][j-1]
		}
	}
	fmt.Println(preSum)
	// Get Sum
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// (1) get idx in mat (2) get idx in dp
			r1 := max(i-k, 0) + 1
			r2 := min(i+k, m-1) + 1
			c1 := max(j-k, 0) + 1
			c2 := min(j+k, n-1) + 1
			// (3) r1 / c1 -= 1
			res[i][j] = preSum[r2][c2] - preSum[r1-1][c2] - preSum[r2][c1-1] + preSum[r1-1][c1-1]
		}
	}
	return res
}

func GetNew2DArray(m int, n int) [][]int {
	res := make([][]int, m)
	for i := 0; i < m; i++ {
		res[i] = make([]int, n)
	}
	return res
}

func max(n1 int, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func min(n1 int, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func GetBlockSum(mat [][]int, i int, j int, k int) int {
	sum := 0
	m, n := len(mat), len(mat[0])
	for r := i - k; r <= i+k; r++ {
		for c := j - k; c <= j+k; c++ {
			if r >= 0 && r < m && c >= 0 && c < n {
				sum += mat[r][c]
			}
		}
	}
	return sum
}

func main() {
	ex := [][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	res := matrixBlockSum(ex, 1)
	fmt.Println(res)
}
