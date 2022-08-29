package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func getAllElements(root1 *TreeNode, root2 *TreeNode) []int {
	sortedList1 := GetSortedList(root1)
	sortedList2 := GetSortedList(root2)
	fmt.Println(sortedList1)
	fmt.Println(sortedList2)
	res := MergeTwoSortedList(sortedList1, sortedList2)
	return res
}

func GetSortedList(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	res := []int{}
	res = append(res, root.Val)
	listL := GetSortedList(root.Left)
	listR := GetSortedList(root.Right)
	res = append(listL, root.Val)
	res = append(res, listR...)
	return res
}

func MergeTwoSortedList(list1 []int, list2 []int) []int {
	l1, l2 := len(list1), len(list2)
	var i, j int
	res := []int{}
	for i < l1 && j < l2 {
		if list1[i] < list2[j] {
			res = append(res, list1[i])
			i++
		} else {
			res = append(res, list2[j])
			j++
		}
	}
	if i == l1 {
		res = append(res, list2[j:]...)
	} else {
		res = append(res, list1[i:]...)
	}
	return res
}

func main() {
	root1 := TreeNode{2, &TreeNode{1, nil, nil}, &TreeNode{4, nil, nil}}
	root2 := TreeNode{1, &TreeNode{0, nil, nil}, &TreeNode{3, nil, nil}}
	res := getAllElements(&root1, &root2)
	fmt.Println(res)
}
