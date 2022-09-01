package main

import "fmt"

/*
	dp[i][j]: max dot product with nums[i - 1] * nums[j - 1] last POSSIBLE dot product
	comes from: (1) nums[i - 1] * nums[j - 1] is last dot product & we can try to add dp[i - 1][j - 1]
				(2) dp[i - 1][j]
				(3) dp[i][j - 1]
*/

func maxDotProduct(nums1 []int, nums2 []int) int {
	m, n := len(nums1), len(nums2)
	dp := NewDp1458(m, n)
	// fmt.Println(dp)
	res := -1000000
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			// include new dot product
			newDotProduct := nums1[i-1] * nums2[j-1]
			dp[i][j] = newDotProduct + max1458(dp[i-1][j-1], 0)
			// do not include new dot product
			if i > 1 {
				dp[i][j] = max1458(dp[i][j], dp[i-1][j])
			}
			if j > 1 {
				dp[i][j] = max1458(dp[i][j], dp[i][j-1])
			}
			if dp[i][j] > res {
				res = dp[i][j]
			}
			fmt.Println(dp[i][j], nums1[i-1], nums2[j-1])
		}
	}
	return res
}

func NewDp1458(m, n int) [][]int {
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	return dp
}

func max1458(n1, n2 int) int {
	if n1 > n2 {
		return n1
	}
	return n2
}

func main() {
	// nums1 := []int{2, 1, -2, 5}
	// nums2 := []int{3, 0, -6}
	nums1 := []int{5, -4, -3}
	nums2 := []int{-4, -3, 0, -4, 2}
	res := maxDotProduct(nums1, nums2)
	fmt.Println(res)
}
