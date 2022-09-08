package main

import (
	"fmt"
	"math"
	"sort"
)

/* Sliding window, size of window == angle */

func visiblePoints(points [][]int, degree int, location []int) int {
	angles := []float64{}
	same := 0
	// Get angle from any point to location
	x0, y0 := location[0], location[1]
	for _, point := range points {
		x, y := point[0], point[1]
		if x == x0 && y == y0 {
			same++
			continue
		}
		fmt.Println(x, y)
		angle := math.Atan2(float64(y-y0), float64(x-x0)) * 180 / math.Pi
		angles = append(angles, angle)
	}
	// To loop through the angles, add all point to back of the array
	n := len(angles)
	for i := 0; i < n; i++ {
		angles = append(angles, angles[i]+360.0)
	}
	sort.Float64s(angles)
	// Get max points with sliding window
	fmt.Println(angles)
	i, j, res := 0, 0, 0
	n = len(angles)
	for j < n {
		for angles[j]-angles[i] > float64(degree) {
			i++
		}
		res = max1617(res, j-i+1)
		j++
	}
	return res + same
}

func max1617(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	points := [][]int{{2, 1}, {2, 2}, {3, 4}, {1, 1}}
	angle := 90
	location := []int{1, 1}
	res := visiblePoints(points, angle, location)
	fmt.Println(res)
}
