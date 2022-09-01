package main

import (
	"fmt"
	"strings"
)

func printVertically(s string) []string {
	words := strings.Split(s, " ")
	// fmt.Println(words)
	maxLength := GetMaxLength(words)
	// fmt.Println(maxLength)
	res := []string{}
	for i := 1; i <= maxLength; i++ {
		// fmt.Println(i)
		var temp string
		for _, word := range words {
			if len(word) >= i {
				temp = temp + string(word[i-1])
			} else {
				temp = temp + " "
			}
		}
		// fmt.Println(temp)
		// clean up all spaces on the right
		for i := len(words) - 1; i >= 0; i-- {
			if string(temp[i]) == " " {
				temp = temp[:i]
			} else {
				break
			}
		}
		res = append(res, temp)
	}
	return res
}

func GetMaxLength(words []string) int {
	maxLength := 0
	for _, word := range words {
		if len(word) > maxLength {
			maxLength = len(word)
		}
	}
	return maxLength
}

func main() {
	ex1 := "TO BE OR NOT TO BE"
	// ex2 := "CONTEST IS COMING"
	res := printVertically(ex1)
	fmt.Println(res)
}
