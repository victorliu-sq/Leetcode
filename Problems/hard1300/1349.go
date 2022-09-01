package main

import (
	"fmt"
	"sort"
	"strconv"
)

func maxStudents(seats [][]byte) int {
	// Get bitmask of each seat
	bitmasks := GetBitmasks(seats)
	// fmt.Println(bitmasks)

	// Get assignment of each seat
	subsets := GetSubsets(seats)
	// fmt.Println(subsets)
	dp := NewDp1349(len(seats), len(seats[0]))
	return dfs1349(0, len(seats), 0, bitmasks, subsets, &dp)
}

func dfs1349(curRow int, m int, lastBitmask int, bitmasks []int, subsets map[int][]int, dp *[]map[int]int) int {
	// base case
	if curRow == m {
		return 0
	}
	// check DP
	if _, ok := (*dp)[curRow][lastBitmask]; ok {
		return (*dp)[curRow][lastBitmask]
	}
	//	recursion next
	res := 0
	for _, bitmaskSeat := range subsets[curRow] {
		// check collision from lastBitMask
		if CheckColCollision(lastBitmask, bitmaskSeat<<1) || CheckColCollision(lastBitmask, bitmaskSeat>>1) {
			continue
		}
		// check collision from neighbor seats -> seats
		if CheckColCollision(bitmaskSeat, bitmaskSeat<<1) || CheckColCollision(bitmaskSeat, bitmaskSeat>>1) {
			continue
		}
		seatsNum := GetOnesNum(bitmaskSeat)
		temp := seatsNum + dfs1349(curRow+1, m, bitmaskSeat, bitmasks, subsets, dp)
		if temp > res {
			res = temp
		}
	}
	// update DP
	(*dp)[curRow][lastBitmask] = res
	return res
}

func GetBitmasks(seats [][]byte) []int {
	bitmasks := []int{}
	for _, seat := range seats {
		bitmask := 0
		n := len(seat)
		for idx, c := range seat {
			if c == '.' {
				bitmask |= 1 << (n - 1 - idx)
			}
		}
		fmt.Println(strconv.FormatInt(int64(bitmask), 2))
	}
	return bitmasks
}

func GetSubsets(seats [][]byte) map[int][]int {
	m, n := len(seats), len(seats[0])
	indices := make([][]int, m)
	// for i := range indices {
	// 	indices[i] = []int{}
	// }
	for i, seat := range seats {
		for j, c := range seat {
			if c == '.' {
				indices[i] = append(indices[i], n-1-j)
			}
		}
		sort.Ints(indices[i])
		fmt.Println(indices[i])
	}
	subsets := map[int][]int{}
	for i := range indices {
		subset := []int{0}
		for _, idx := range indices[i] {
			newBitmasks := []int{}
			for _, bitmask := range subset {
				newBitmask := bitmask | (1 << idx)
				newBitmasks = append(newBitmasks, newBitmask)
			}
			subset = append(subset, newBitmasks...)
		}
		subsets[i] = subset
		// fmt.Println(subset)
	}
	return subsets
}

func CheckColCollision(bitmask1 int, bitmask2 int) bool {
	// if & of two nums are not 0, there must be same columns
	if bitmask1&bitmask2 != 0 {
		return true
	}
	return false
}

func GetOnesNum(num int) int {
	res := 0
	for num > 0 {
		if num%2 == 1 {
			res++
		}
		num = num / 2
	}
	return res
}

func NewDp1349(m, n int) []map[int]int {
	dp := make([]map[int]int, m)
	for i := range dp {
		dp[i] = make(map[int]int)
	}
	return dp
}

func main() {
	seats := [][]byte{{'#', '.', '#', '#', '.', '#'}, {'.', '#', '#', '#', '#', '.'}, {'#', '.', '#', '#', '.', '#'}}
	res := maxStudents(seats)
	fmt.Println(res)
}
