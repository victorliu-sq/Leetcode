package main

func removeLeafNodes(root *TreeNode, target int) *TreeNode {
	var canDelete bool
	canDelete = false
	dfs1325(root, target, &canDelete)
	for canDelete {
		canDelete = false
		root = dfs1325(root, target, &canDelete)
	}
	return root
}

func dfs1325(node *TreeNode, target int, canDelete *bool) *TreeNode {
	if node == nil {
		return nil
	}
	if node.Left == nil && node.Right == nil && node.Val == target {
		*canDelete = true
		return nil
	}
	node.Left = dfs1325(node.Left, target, canDelete)
	node.Right = dfs1325(node.Right, target, canDelete)
	return node
}

func main() {

}
