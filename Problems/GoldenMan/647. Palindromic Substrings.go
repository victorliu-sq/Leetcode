func countSubstrings(s string) int {
	// initialize dp
	n := len(s)
	dp := make([][]bool, n)
	res := 0
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}

	// initialize dp with palindrome whose length == 1 or length == 2
	for i := 0; i < n; i++ {
		dp[i][i] = true
		res++
	}
	for i := 0; i < n-1; i++ {
		if s[i] == s[i+1] {
			dp[i][i+1] = true
			res++
		}
	}
	// iterate length from 3 to n
	for l := 3; l <= n; l++ {
		// n - 1 - i + 1 = l -> i = n -l
		for i := 0; i <= n-l; i++ {
			j := i + l - 1
			if dp[i+1][j-1] && s[i] == s[j] {
				dp[i][j] = true
				res++
			}
		}
	}
	return res
}