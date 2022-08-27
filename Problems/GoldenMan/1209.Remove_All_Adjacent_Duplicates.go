type Node struct {
	Val   rune
	Count int
}

func newNode(val rune, count int) *Node {
	node := Node{}
	node.Val = val
	node.Count = count
	return &node
}

func removeDuplicates(s string, k int) string {
	stack := []*Node{}
	for _, c := range s {
		if len(stack) == 0 || stack[len(stack)-1].Val != c {
			node := newNode(c, 1)
			stack = append(stack, node)
		} else {
			stack[len(stack)-1].Count += 1
			if stack[len(stack)-1].Count == k {
				stack = stack[:len(stack)-1]
			}
		}
	}

	res := []rune{}
	for _, node := range stack {
		for i := 0; i < node.Count; i += 1 {
			res = append(res, node.Val)
		}
	}
	return string(res)
	// fmt.Println(s)
}