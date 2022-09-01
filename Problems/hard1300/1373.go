package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func maxSumBST(root *TreeNode) int {
	var res int
	dfs1373(root, &res)
	return res
}

func dfs1373(node *TreeNode, res *int) []int {
	// base case
	if node == nil {
		return []int{100000, -100000, 0, 1}
	}
	// recursion next
	// return value{min, max, sum, isBinary}, not binary tree
	resL, resR := dfs1373(node.Left, res), dfs1373(node.Right, res)
	fmt.Println(resL, resR)
	minL, maxL, sumL, isBinaryL := resL[0], resL[1], resL[2], resL[3]
	minR, maxR, sumR, isBinaryR := resR[0], resR[1], resR[2], resR[3]
	// (1) check whether binary
	if isBinaryL > 0 && isBinaryR > 0 && maxL < node.Val && node.Val < minR {
		newSum := sumL + sumR + node.Val
		*res = max1373(newSum, *res)
		// leave node
		if minL == 100000 {
			minL = node.Val
		}
		if maxR == -100000 {
			maxR = node.Val
		}
		return []int{minL, maxR, newSum, 1}
	}
	// (2) not binary
	return []int{0, 0, 0, 0}
}

func max1373(n1, n2 int) int {
	if n1 < n2 {
		return n2
	}
	return n1
}

func main() {
}
