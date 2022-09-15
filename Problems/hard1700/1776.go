package main

import (
	"container/list"
	"fmt"
)

func getCollisionTimes(cars [][]int) []float64 {
	n := len(cars)
	// Initialize times
	times := make([]float64, n)
	for i := 0; i < n; i++ {
		times[i] = -1
	}
	stack := list.New()
	for i := n - 1; i >= 0; i-- {
		pos, speed := cars[i][0], cars[i][1]
		// Pop out all cars that current car cannot catch p :
		// (1) speed < last car's (2) speed not enough to catch up car fleet before last car
		for stack.Len() > 0 && !CanCatchUp(pos, speed, stack, cars, times) {
			lastCarIdxP := stack.Back()
			stack.Remove(lastCarIdxP)
		}
		// Get time of current car catching up last car fleet
		if stack.Len() > 0 {
			lastCarIdx := stack.Back().Value.(int)
			lastPos, lastSpeed := cars[lastCarIdx][0], cars[lastCarIdx][1]
			times[i] = GetCatchUpTime(pos, speed, lastPos, lastSpeed)
		}
		// Add current idx to stack
		stack.PushBack(i)
		// fmt.Println("stack values")
		// for e := stack.Front(); e != nil; e = e.Next() {
		// 	fmt.Println(e.Value.(int))
		// }
	}
	return times
}

func CanCatchUp(pos, speed int, stack *list.List, cars [][]int, times []float64) bool {
	lastCarIdx := stack.Back().Value.(int)
	lastPos, lastSpeed := cars[lastCarIdx][0], cars[lastCarIdx][1]
	if speed <= lastSpeed {
		return false
	}
	time := GetCatchUpTime(pos, speed, lastPos, lastSpeed)
	if times[lastCarIdx] > 0 && times[lastCarIdx] < time {
		return false
	}
	return true
}

func GetCatchUpTime(pos, speed, lastPos, lastSpeed int) float64 {
	return float64(lastPos-pos) / float64(speed-lastSpeed)
}

func main() {
	cars := [][]int{{1, 2}, {2, 1}, {4, 3}, {7, 2}}
	res := getCollisionTimes(cars)
	fmt.Println(res)
}
