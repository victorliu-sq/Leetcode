package main

import (
	"fmt"
)

/*
Each time find smallest num within distance k, swap pairs of nums all the way to the smallest num
*/

func minInteger(num string, k int) string {
	curIdx := 0
	curSwapTimes := 0
	n := len(num)
	fmt.Println(num)
	for curSwapTimes < k && curIdx < n {
		start := curIdx
		// fmt.Println("curSwapTimes:", curSwapTimes)
		end := min1505(n-1, curIdx+k-curSwapTimes)
		minIdx := curIdx
		minDigit := num[curIdx]
		for i := start + 1; i <= end; i++ {
			if IsSmaller(num[i], minDigit) {
				minDigit = num[i]
				minIdx = i
			}
		}
		fmt.Println(minIdx, string(num[minIdx]))
		for j := minIdx; j >= curIdx+1; j-- {
			i := j - 1
			num = SwapNum(i, j, num)
			fmt.Println(num)
		}
		curSwapTimes += minIdx - curIdx
		curIdx += 1
	}
	return num
}

func IsSmaller(c1, c2 byte) bool {
	if int(c1) < int(c2) {
		return true
	}
	return false
}

// To swap 2 chars in 1 string, we need to convert string into rune array then convert it back to string
func SwapNum(i, j int, num string) string {
	rune := []byte(num)
	rune[i], rune[j] = rune[j], rune[i]
	return string(rune)
}

func min1505(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func main() {
	num := "4321"
	k := 4
	res := minInteger(num, k)
	fmt.Println(res)
}
