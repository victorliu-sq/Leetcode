package main

// type TreeNode struct {
// 	Val   int
// 	Left  *TreeNode
// 	Right *TreeNode
// }

func sumEvenGrandparent(root *TreeNode) int {
	var res int
	dfsSumEvenGrandParent(root, nil, nil, &res)
	return res
}

func dfsSumEvenGrandParent(node *TreeNode, nodeP *TreeNode, nodeG *TreeNode, res *int) {
	if node == nil {
		return
	}
	if nodeG != nil && nodeG.Val%2 == 0 {
		*res += node.Val
	}
	dfsSumEvenGrandParent(node.Left, node, nodeP, res)
	dfsSumEvenGrandParent(node.Right, node, nodeP, res)
}

// func main() {
// 	res := sumEvenGrandparent(&TreeNode{6, &TreeNode{7, &TreeNode{2, &TreeNode{9}, nil}, }})
// 	fmt.Println(res)
// }
