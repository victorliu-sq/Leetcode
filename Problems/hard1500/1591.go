package main

import "fmt"

/*
	instead of trying to paint color from 0, we try to erase color into 0

	each time, we try to erase a whole rectangle of colors into 0

	if all colors in this rectangle is either one color or 0, we can erase this rectangle
*/

func isPrintable(targetGrid [][]int) bool {
	rectangles := GetRectangles(targetGrid)
	// fmt.Println(rectangles)
	colors := GetColors(rectangles)
	// fmt.Println(colors)
	// try to erase colors as many as possible
	for len(colors) > 0 {
		newColors := []int{}
		for _, color := range colors {
			if !CanErase(targetGrid, rectangles, color) {
				newColors = append(newColors, color)
				continue
			}
			Erase(targetGrid, rectangles, color)
		}
		// fmt.Println(targetGrid, newColors)
		if len(colors) == len(newColors) {
			return false
		}
		colors = newColors
	}
	return true
}

func CanErase(grid [][]int, rectangles map[int][]int, color int) bool {
	r1, c1, r2, c2 := rectangles[color][0], rectangles[color][1], rectangles[color][2], rectangles[color][3]
	for i := r1; i <= r2; i++ {
		for j := c1; j <= c2; j++ {
			if grid[i][j] != color && grid[i][j] != 0 {
				return false
			}
		}
	}
	return true
}

func Erase(grid [][]int, rectangles map[int][]int, color int) {
	r1, c1, r2, c2 := rectangles[color][0], rectangles[color][1], rectangles[color][2], rectangles[color][3]
	for i := r1; i <= r2; i++ {
		for j := c1; j <= c2; j++ {
			grid[i][j] = 0
		}
	}
}

func GetRectangles(grid [][]int) map[int][]int {
	rectangles := map[int][]int{}
	for i := range grid {
		for j := range grid[i] {
			color := grid[i][j]
			if _, ok := rectangles[color]; !ok {
				rectangles[color] = []int{i, j, i, j}
			} else {
				rectangles[color][0] = min1591(rectangles[color][0], i)
				rectangles[color][1] = min1591(rectangles[color][1], j)
				rectangles[color][2] = max1591(rectangles[color][2], i)
				rectangles[color][3] = max1591(rectangles[color][3], j)
			}
		}
	}
	return rectangles
}

func GetColors(rectangles map[int][]int) []int {
	colors := []int{}
	for color, _ := range rectangles {
		colors = append(colors, color)
	}
	return colors
}

func min1591(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func max1591(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	targetGrid := [][]int{{1, 1, 1, 1}, {1, 1, 3, 3}, {1, 1, 3, 4}, {5, 5, 1, 4}}
	res := isPrintable(targetGrid)
	fmt.Println(res)
}
