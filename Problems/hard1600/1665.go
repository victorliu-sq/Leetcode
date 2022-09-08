package main

import (
	"fmt"
	"sort"
)

/*
	Assume we are going to finish tasks one by one,
	for each task [actual, min], we can save (min - actual) energy
	before we start a new task, we want to save as much energy as possible
*/

func minimumEffort(tasks [][]int) int {
	// fmt.Println(tasks)
	// 1. sort tasks by saved energy
	newTasks := [][]int{}
	for _, task := range tasks {
		actual, min := task[0], task[1]
		task = append(task, min-actual)
		// fmt.Println(task)
		newTasks = append(newTasks, task)
	}
	sort.Slice(newTasks, func(i, j int) bool {
		if newTasks[i][2] >= newTasks[j][2] {
			return true
		}
		return false
	})
	// fmt.Println(newTasks)
	// 2.iterate each task and update cost & curEnergy
	totalCost, curEnergy := 0, 0
	for _, task := range newTasks {
		actual, min := task[0], task[1]
		if curEnergy < min {
			// if not enough energy, add up to min and consume some
			totalCost += min - curEnergy
			curEnergy = min - actual
		} else {
			// if enough energy, just consume some energy
			curEnergy -= actual
		}
	}
	return totalCost
}

func main() {
	tasks := [][]int{{1, 2}, {2, 4}, {4, 8}}
	res := minimumEffort(tasks)
	fmt.Println(res)
}
