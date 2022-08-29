package main

import "fmt"

func maxProduct(root *TreeNode) int {
	sum := 0
	// Get sum of all nodes
	GetSum(root, &sum)
	maxP := 0
	// Get maxProduct = (sum of 1 subtree) * (sum of all nodes - sum of 1 subtree)
	GetMaxProduct(root, sum, &maxP)
	return maxP
}

func GetSum(node *TreeNode, sum *int) {
	if node == nil {
		return
	}
	*sum += node.Val
	GetSum(node.Left, sum)
	GetSum(node.Right, sum)
}

func GetMaxProduct(node *TreeNode, sum int, maxP *int) int {
	if node == nil {
		return 0
	}
	sumL := GetMaxProduct(node.Left, sum, maxP)
	sumR := GetMaxProduct(node.Right, sum, maxP)
	*maxP = max(*maxP, max(sumL*(sum-sumL), sumR*(sum-sumR)))
	fmt.Println(*maxP)
	return sumL + sumR + node.Val
}
