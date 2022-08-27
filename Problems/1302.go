package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func deepestLeavesSum(root *TreeNode) int {
	var deepestDepth int
	// pass slice by reference
	deepestNodes := []*TreeNode{}
	dfs(root, 1, &deepestNodes, &deepestDepth)
	res := 0
	for _, treeNode := range deepestNodes {
		res += treeNode.Val
	}
	return res
}

func dfs(node *TreeNode, curDepth int, deepestNodes *[]*TreeNode, deepestDepth *int) {
	if node == nil {
		return
	}
	if curDepth > *deepestDepth {
		*deepestDepth = curDepth
		*deepestNodes = []*TreeNode{}
		*deepestNodes = append(*deepestNodes, node)
		// fmt.Println(deepestNode)
	} else if curDepth == *deepestDepth {
		*deepestNodes = append(*deepestNodes, node)
		// fmt.Println(deepestNode)
	}
	dfs(node.Left, curDepth+1, deepestNodes, deepestDepth)
	dfs(node.Right, curDepth+1, deepestNodes, deepestDepth)
}

func main() {
	var root *TreeNode
	// *root := [1,2,3,4,5,null,6,7,null,null,null,null,8]
	res := deepestLeavesSum()
	fmt.Println(res)
}
