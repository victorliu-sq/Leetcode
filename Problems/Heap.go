// min Heap
type MinHeap []Team

func (h *MinHeap) Len() int {
	return len(*h)
}

func (h *MinHeap) Less(i, j int) bool {
	return (*h)[i].Speed <= (*h)[j].Speed
}

func (h *MinHeap) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

// we must use interface{} instead of int
func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(Team))
}

func (h *MinHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	(*h) = (*h)[:len(*h)-1]
	return x
}

// ************************************************************
// max Heap
type MinHeap []Team

func (h *MinHeap) Len() int {
	return len(*h)
}

func (h *MinHeap) Less(i, j int) bool {
	return (*h)[i].Speed >= (*h)[j].Speed
}

func (h *MinHeap) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

// we must use interface{} instead of int
func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(Team))
}

func (h *MinHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	(*h) = (*h)[:len(*h)-1]
	return x
}
