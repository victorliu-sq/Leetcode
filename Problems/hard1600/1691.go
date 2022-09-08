package main

import (
	"fmt"
	"sort"
)

/*
"You can place cuboid i on cuboid j if width[i] <= width[j] and length[i] <= length[j] and height[i] <= height[j]"
That's much easier.
*/

func maxHeight(cuboids [][]int) int {
	// sort each cuboids from small to big (always height is biggest one)
	for _, cuboid := range cuboids {
		sort.Ints(cuboid)
	}
	fmt.Println(cuboids)
	// sort cuboids from big to small
	sort.Slice(cuboids, func(i, j int) bool {
		if cuboids[i][0] != cuboids[j][0] {
			return cuboids[i][0] > cuboids[j][0]
		}
		if cuboids[i][1] != cuboids[j][1] {
			return cuboids[i][1] > cuboids[j][1]
		}
		return cuboids[i][2] > cuboids[j][2]
	})
	fmt.Println(cuboids)
	// initialize dp
	dp := GetDp(cuboids)
	fmt.Println(dp)
	n := len(cuboids)
	res := 0
	// for each cuboid, check whether it can stand on a previous cuboid
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			if CanContain(cuboids, j, i) {
				// fmt.Println(j, i)
				dp[i] = max1691(dp[i], dp[j]+cuboids[i][2])
			}
		}
		res = max1691(res, dp[i])
	}
	return res
}

func max1691(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func GetDp(cuboids [][]int) []int {
	n := len(cuboids)
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = cuboids[i][2]
	}
	return dp
}

func CanContain(cuboids [][]int, j, i int) bool {
	if cuboids[j][0] >= cuboids[i][0] && cuboids[j][1] >= cuboids[i][1] && cuboids[j][2] >= cuboids[i][2] {
		return true
	}
	return false
}

func main() {
	cuboids := [][]int{{50, 45, 20}, {95, 37, 53}, {45, 23, 12}}
	res := maxHeight(cuboids)
	fmt.Println(res)
}
