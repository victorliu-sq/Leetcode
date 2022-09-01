package main

import (
	"fmt"
	"sort"
)

func filterRestaurants(restaurants [][]int, veganFriendly int, maxPrice int, maxDistance int) []int {
	candidates := [][]int{}
	for _, restaurant := range restaurants {
		// id, rating, vegan, prices, distance
		_, _, curVeganFriendly, curPrice, curDistance := restaurant[0], restaurant[1], restaurant[2], restaurant[3], restaurant[4]
		// if veganFriendly == 1, must equal
		if veganFriendly == 1 {
			if curVeganFriendly == veganFriendly && curPrice <= maxPrice && curDistance <= maxDistance {
				candidates = append(candidates, restaurant)
			}
			// if veganFriendly == 0, do not consider
		} else {
			if curPrice <= maxPrice && curDistance <= maxDistance {
				candidates = append(candidates, restaurant)
			}
		}
	}
	// fmt.Println(candidates)
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i][1] == candidates[j][1] {
			return candidates[i][0] > candidates[j][0]
		}
		return candidates[i][1] > candidates[j][1]
	})
	res := []int{}
	for _, candidate := range candidates {
		res = append(res, candidate[0])
	}
	return res
}

func main() {
	restaurants := [][]int{{1, 4, 1, 40, 10}, {2, 8, 0, 50, 5}, {3, 8, 1, 30, 4}, {4, 10, 0, 10, 3}, {5, 1, 1, 15, 1}}
	veganFriendly := 1
	maxPrice := 50
	maxDistance := 10
	res := filterRestaurants(restaurants, veganFriendly, maxPrice, maxDistance)
	fmt.Println(res)
}
