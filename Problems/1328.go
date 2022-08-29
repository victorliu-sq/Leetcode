func breakPalindrome(palindrome string) string {
	if len(palindrome) == 1 {
		return ""
	}
	arr := []rune(palindrome)
	res := []rune{}
	// try to convert 1 letter of left part to 'a' at first
	for i := 0; i < len(palindrome)/2; i++ {
		c := arr[i]
		// fmt.Println(c)
		if c == 'a' {
			res = append(res, c)
		} else {
			res = append(res, arr[i:]...)
			res[i] = 'a'
			return string(res)
		}
	}
	// if all left parts are 'a', we only need to convert last letter to 'b'
	arr[len(arr)-1] = 'b'
	return string(arr)
}