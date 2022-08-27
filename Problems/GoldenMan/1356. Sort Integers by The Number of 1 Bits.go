func sortByBits(arr []int) []int {
	sort.Slice(arr, func(i, j int) bool {
		onesI, onesJ := bits.OnesCount(uint(arr[i])), bits.OnesCount(uint(arr[j]))
		if onesI != onesJ {
			return onesI < onesJ
		} else {
			return arr[i] < arr[j]
		}
	})
	return arr
}