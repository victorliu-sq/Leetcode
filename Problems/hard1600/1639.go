package main

import "fmt"

func numWays(words []string, target string) int {
	// count -> count times of char c at index i
	counts := GetCount(words)
	// fmt.Println(counts)
	dp := map[int]int{}
	res := dfs1639(0, 0, len(words[0]), target, counts, dp)
	return res
}

func dfs1639(i, j, length int, target string, counts [][]int, dp map[int]int) int {
	// Base Case
	if i == len(target) {
		return 1
	}
	if j == length {
		return 0
	}
	// Check DP
	key := i<<10 + j
	if _, ok := dp[key]; ok {
		return dp[key]
	}
	// Next Recursion
	// (1) utilize jth char
	res := 0
	res += dfs1639(i, j+1, length, target, counts, dp)
	res %= 1000000007
	// (2) do not utilize jth char
	char := target[i] - 'a'
	if j <= len(counts[char])-1 && counts[char][j] > 0 {
		res += counts[char][j] * dfs1639(i+1, j+1, length, target, counts, dp)
		res %= 1000000007
	}
	// for idx, count := range counts[target[i]-'a'] {
	// 	// fmt.Printf("current char is %vth char %v, idx is %v and count is %v\n", i, string(target[i]), idx, count)
	// 	if idx >= j {
	// 		res += count * dfs1639(i+1, idx+1, length, target, counts, dp)
	// 		res %= 1000000007
	// 	}
	// }
	dp[key] = res
	return res
}

func GetCount(words []string) [][]int {
	length := len(words[0])
	count := make([][]int, 26)
	for i := range count {
		count[i] = make([]int, length)
	}
	for _, word := range words {
		for idx := range word {
			count[word[idx]-'a'][idx] += 1
		}
	}
	// fmt.Println(count)
	return count
}

func main() {
	words := []string{"acca", "bbbb", "caca"}
	target := "aba"
	res := numWays(words, target)
	fmt.Println(res)
}
