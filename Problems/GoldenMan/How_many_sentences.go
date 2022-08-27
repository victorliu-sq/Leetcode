package main

import (
	"fmt"
	"sort"
	"strings"
)

/* Given an array of words and an array of sentences, determine which words are anagrams of each other.
Calculate how many sentences can be created by replacing any word with one of its anagrams,
Example wordSet = ['listen' 'silent, 'it', 'is'] sentence = "listen it is silent"
Determine that listen is an anagram of silent. Those two words can be replaced with their anagrams.
 The four sentences that can be created are:
 • listen it is silent • listen it is listen • silent it is silent • silent it is listen​ */

func HowManySentences(words []string, sentences string) int {
	// (1) for each word, find its least lexicographical anagram, count of anagram += 1
	count := make(map[string]int)
	for _, word := range words {
		key := LeastLexicographicalAnagram(word)
		count[key] += 1
		// fmt.Println("word: %v, count, %v", word, count[key])
	}

	res := 0
	split := strings.Split(sentences, " ")
	// (2) for each word in sentences, if current word has anagram, res += count
	for _, curW := range split {
		// fmt.Println(curW)
		key := LeastLexicographicalAnagram(curW)
		if count[key] > 1 {
			res += count[key]
		}
	}
	return res
}

type RuneSlice []rune

func LeastLexicographicalAnagram(s string) string {
	// (1) convert string into rune slice
	temp := []rune(s)
	// (2) sort the rune slice
	sort.Slice(temp, func(i, j int) bool {
		return temp[i] < temp[j]
	})
	// (3) convert the rune slice back into string
	return string(temp)
}

func main() {
	words := []string{"listen", "silent", "it", "is"}
	sentences := "listen it is silent"
	res := HowManySentences(words, sentences)
	fmt.Println(res)
}
