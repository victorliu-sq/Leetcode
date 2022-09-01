package main

import (
	"fmt"
	"strconv"
)

func numberWays(hats [][]int) int {
	// return 0
	people := GetPeople(hats)
	peopleNum := len(hats)
	// fmt.Println(people)
	dp := map[int64]int{}
	return dfs1434(1, 41, peopleNum, 0, people, dp)
}

func GetPeople(hats [][]int) [][]int {
	people := make([][]int, 41)
	for i := range people {
		people[i] = []int{}
	}
	for i := range hats {
		for _, hat := range hats[i] {
			people[hat] = append(people[hat], i)
		}
	}
	return people
}

func dfs1434(curIdx, n, peopleNum int, bitmask int64, people [][]int, dp map[int64]int) int {
	// fmt.Println(curIdx)
	// base case
	// (1) all people have been assigned
	if CheckAllPeopleAssigned(bitmask, peopleNum) {
		fmt.Println(curIdx, strconv.FormatInt(bitmask, 2))
		return 1
	}
	// (2) all hats have been assigned
	if curIdx == n {
		return 0
	}
	// check dp
	key := bitmask<<6 + int64(curIdx)
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// next recursion
	res := 0
	// assign one person
	for _, person := range people[curIdx] {
		// check used
		if bitmask&(1<<int64(person)) != 0 {
			continue
		}
		newBitmask := bitmask | (1 << int64(person))
		res += dfs1434(curIdx+1, n, peopleNum, newBitmask, people, dp)
		res %= 1000000007
	}
	// do not assign any person
	res += dfs1434(curIdx+1, n, peopleNum, bitmask, people, dp)
	res %= 1000000007
	// update dp
	dp[key] = res
	return res
}

func CheckAllPeopleAssigned(bitmask int64, peopleNum int) bool {
	var allMask int64
	for i := 0; i < peopleNum; i++ {
		allMask |= (1 << int64(i))
	}
	// fmt.Println(allMask)
	if (allMask & bitmask) == allMask {
		return true
	}
	return false
}

/*
too many hats but people are not that much
assign people with hats => assign hats with people
*/
/* func numberWays(hats [][]int) int {
	// return 0
	dp := map[int64]int{}
	return dfs1434(0, len(hats), 0, hats, dp)
}

func dfs1434(curIdx, n int, bitmask int64, hats [][]int, dp map[int64]int) int {
	if curIdx == n {
		return 1
	}
	if _, ok := dp[bitmask]; ok {
		return dp[bitmask]
	}
	res := 0
	for _, hat := range hats[curIdx] {
		// check used
		if bitmask&(1<<int64(hat)) != 0 {
			continue
		}
		newBitmask := bitmask | (1 << int64(hat))
		res += dfs1434(curIdx+1, n, newBitmask, hats, dp)
	}
	dp[bitmask] = res
	return res
} */

func main() {
	// hats := [][]int{{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}}
	hats := [][]int{{3, 5, 1}, {3, 5}}
	res := numberWays(hats)
	fmt.Println(res)
}
