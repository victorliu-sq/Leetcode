package main

import (
	"fmt"
	"math"
	"sort"
)

/*
Time Complexity: iteration ^ depth -> O(10 ^ 26)
Space Complexity: depth -> O(26)
*/

func isSolvable(words []string, result string) bool {
	coefficient := map[byte]int{}
	// for each char of each word, get its coefficient -> positive
	for i := range words {
		for j := range words[i] {
			char := words[i][j]
			coefficient[char] += GetCoefficient(words[i], j)
			// fmt.Println(string(char), coefficient[char])
		}
	}

	// for each char of result, get its coefficient => negative
	for i := range result {
		char := result[i]
		coefficient[char] -= GetCoefficient(result, i)
		// fmt.Println(string(char), coefficient[char])
	}
	// fmt.Println(coefficient)
	// for each leading char in words and result, set its as nonzero
	nonzero := map[byte]bool{}
	for i := range words {
		if len(words[i]) > 1 {
			nonzero[words[i][0]] = true
		}
	}
	if len(result) > 1 {
		nonzero[result[0]] = true
	}
	fmt.Println(nonzero)
	// assign 1 num for each char using dfs + bitmask
	keys := []byte{}
	for c, _ := range coefficient {
		keys = append(keys, c)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i] < keys[j] {
			return true
		}
		return false
	})

	// fmt.Println(keys)
	return dfs(0, keys, 0, 0, coefficient, nonzero)
}

func dfs(idx int, keys []byte, bitmaskNum int, sum int, coefficient map[byte]int, nonzero map[byte]bool) bool {
	// base case
	if idx == len(keys) {
		// fmt.Println(sum, strconv.FormatInt(int64(bitmaskNum), 2))
		return sum == 0
	}
	// for each num, try to assign it to cur char
	res := false
	key := keys[idx]
	// fmt.Println(idx, strconv.FormatInt(int64(bitmaskNum), 2))
	for num := 0; num < 10; num++ {
		//used
		if (1<<num)&bitmaskNum != 0 {
			continue
		}
		//non-leading zero
		if _, ok := nonzero[key]; num == 0 && ok {
			continue
		}
		newIdx := idx + 1
		newBitmaskNum := bitmaskNum | (1 << num)
		newSum := sum + num*coefficient[key]

		res = res || dfs(newIdx, keys, newBitmaskNum, newSum, coefficient, nonzero)
		if res {
			return res
		}
	}
	// recursion next char
	return res
}

func GetCoefficient(word string, idx int) int {
	pos := len(word) - 1 - idx
	return int(math.Pow(10, float64(pos)))
}

func main() {
	words := []string{"A", "B"}
	result := "A"
	res := isSolvable(words, result)
	fmt.Println(res)
}
