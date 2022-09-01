package main

import "fmt"

/*
s : abcdeffedcba
prefix ->  <- suffix
get a number instead of string to fasten speed
*/

func longestPrefix(s string) string {
	n := len(s)
	// mod := 100000007
	maxLength := 0
	prefix := 0
	suffix := 0
	// the coefficient does not need to be too large
	suffixCoefficient := 1
	// except itself
	for i := 0; i < n-1; i++ {
		num1 := int(s[0+i] - 'a' + 1)
		num2 := int(s[n-1-i] - 'a' + 1)
		prefix = (prefix*2 + num1)
		suffix = (num2*suffixCoefficient + suffix)
		suffixCoefficient *= 2
		// fmt.Println(prefix, suffix)
		if prefix == suffix {
			// fmt.Println(s[:i+1], s[n-i-1:])
			temp := i + 1
			if temp > maxLength {
				maxLength = temp
			}
		}
	}
	return s[:maxLength]
}

func main() {
	s := "dccbcdbbcbaabcdabbbadaaaacaabcddbaccdccbcdbbcbaabcdabbbadaa"
	res := longestPrefix(s)
	fmt.Println(res)
}
