package main

import "fmt"

func minFlips(a int, b int, c int) int {
	res := 0
	for a > 0 || b > 0 || c > 0 {
		aBit, bBit, cBit := GetLastBit(a), GetLastBit(b), GetLastBit(c)
		if aBit|bBit == cBit {
			// if no need to change
			res += 0
		} else if aBit == 1 && bBit == 1 && cBit == 0 {
			// if 1, 1 => 0, change both
			res += 2
		} else {
			// else only change 1 bit
			res += 1
		}
		a, b, c = a/2, b/2, c/2
	}
	return res
}

func GetLastBit(num int) int {
	return num % 2
}

func main() {
	a, b, c := 2, 6, 5
	res := minFlips(a, b, c)
	fmt.Println(res)
}
