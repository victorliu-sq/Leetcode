func deleteDuplicatesUnsorted(head *ListNode) *ListNode {
	visited := make(map[int]bool)
	duplicated := make(map[int]bool)
	it := head
	for it != nil {
		if _, ok := visited[it.Val]; !ok {
			visited[it.Val] = true
		} else {
			duplicated[it.Val] = true
		}
		it = it.Next
	}

	dummy := ListNode{}
	dummy.Next = head
	it = &dummy
	for it != nil && it.Next != nil {
		if _, ok := duplicated[it.Next.Val]; ok {
			it.Next = it.Next.Next
		} else {
			it = it.Next
		}
	}
	return dummy.Next
}