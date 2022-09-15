package main

import (
	"fmt"
)

/*
End can be visited for multiple times, we need to use fuel to mark the end as failed path

*/

func countRoutes(locations []int, start int, finish int, fuel int) int {
	dp := map[int]int{}
	res := dfs1575(start, finish, fuel, []int{start}, locations, dp)
	return res
}

func dfs1575(curCity, endCity, fuel int, path, locations []int, dp map[int]int) int {
	if fuel < 0 {
		fmt.Println(path)
		return 0
		// finish can be visited multiple times
	}
	key := GetKey(curCity, fuel)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// at first, res == 1 if curCity == endCity
	res := 0
	if curCity == endCity {
		res++
	}
	n := len(locations)
	for i := 0; i < n; i++ {
		if i == curCity {
			continue
		}
		// fmt.Println(curCity)
		curPos := locations[curCity]
		newPos := locations[i]
		// fmt.Println(curCity, i, GetDifference(curPos, newPos), fuel)
		newFuel := fuel - GetDifference(curPos, newPos)
		newPath := append(path, i)
		res += dfs1575(i, endCity, newFuel, newPath, locations, dp)
		res %= 1000000007
	}
	dp[key] = res
	return res
}

func GetKey(curCity, fuel int) int {
	// curCity -> 100 -> 7
	// fuel -> 200 -> 8
	key := curCity + fuel<<7
	return key
}

func GetDifference(pos1, pos2 int) int {
	if pos1 > pos2 {
		return pos1 - pos2
	}
	return pos2 - pos1
}

func main() {
	// locations := []int{2, 3, 6, 8, 4}
	// start := 1
	// finish := 3
	// fuel := 5
	locations := []int{4, 3, 1}
	start := 1
	finish := 0
	fuel := 6
	res := countRoutes(locations, start, finish, fuel)
	fmt.Println(res)
}
