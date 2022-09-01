package main

import (
	"fmt"
	"sort"
)

func diagonalSort(mat [][]int) [][]int {
	// SortDiagonalElements(&mat, startX, startY)
	m, n := len(mat), len(mat[0])
	for i := 0; i < m; i++ {
		SortDiagonalElements(&mat, i, 0)
	}
	for j := 1; j < n; j++ {
		SortDiagonalElements(&mat, 0, j)
	}
	return mat
}

func SortDiagonalElements(mat *[][]int, startX int, startY int) {
	m, n := len(*mat), len((*mat)[0])
	var length int
	if m-startX < n-startY {
		length = m - startX
	} else {
		length = n - startY
	}
	elements := []int{}
	for i := 0; i < length; i++ {
		elements = append(elements, (*mat)[startX+i][startY+i])
	}
	// fmt.Println(elements)
	sort.Ints(elements)
	// fmt.Println(elements)
	for i := 0; i < length; i++ {
		(*mat)[startX+i][startY+i] = elements[i]
	}
}

func main() {
	ex := [][]int{{3, 3, 1, 1}, {2, 2, 1, 2}, {1, 1, 1, 2}}
	res := diagonalSort(ex)
	fmt.Println(res)
}
