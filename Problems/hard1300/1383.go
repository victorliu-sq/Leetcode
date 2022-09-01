package main

import (
	"container/heap"
	"fmt"
	"sort"
)

/*
Given a set of teams, performance = min efficiency * sum of speeds
In this way, we can fix efficiency and get sum of speeds from teams whose efficiency >=  it(e1)
	[bigger efficiency] e1
	if we move to smaller efficiency e2, speed of e1 can be candidates
	[bigger efficiency, e1 ] e2
*/

func maxPerformance(n int, speed []int, efficiency []int, k int) int {
	teams := GetTeamArray(n, speed, efficiency)
	// sort team by efficiency from big to small
	sort.Slice(teams, func(i, j int) bool {
		if teams[i].Efficiency >= teams[j].Efficiency {
			return true
		}
		return false
	})
	fmt.Println(teams)
	h := &MinHeap1383{}
	res := 0
	totalSpeed := 0
	// iterate team by efficiency from big to small
	for _, team := range teams {
		// if candidates are full, move one with min speed
		if h.Len() == k {
			minSpeed := heap.Pop(h).(Team).Speed
			totalSpeed -= minSpeed
		}
		// add current node to group
		heap.Push(h, team)
		// min efficiency = team.efficiency
		totalSpeed += team.Speed
		temp := totalSpeed * team.Efficiency
		if temp > res {
			res = temp
		}
	}
	return res
}

func GetTeamArray(n int, speed []int, efficiency []int) []Team {
	teams := []Team{}
	for i := 0; i < n; i++ {
		teams = append(teams, Team{speed[i], efficiency[i]})
	}
	return teams
}

func main() {
	n := 6
	speed := []int{2, 10, 3, 1, 5, 8}
	efficiency := []int{5, 4, 3, 9, 7, 2}
	k := 2
	res := maxPerformance(n, speed, efficiency, k)
	fmt.Println(res)
}

// minHeap
type Team struct {
	Speed      int
	Efficiency int
}

type MinHeap1383 []Team

func (h *MinHeap1383) Len() int {
	return len(*h)
}

func (h *MinHeap1383) Less(i, j int) bool {
	return (*h)[i].Speed <= (*h)[j].Speed
}

func (h *MinHeap1383) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

// we must use interface{} instead of int
func (h *MinHeap1383) Push(x interface{}) {
	*h = append(*h, x.(Team))
}

func (h *MinHeap1383) Pop() interface{} {
	x := (*h)[len(*h)-1]
	(*h) = (*h)[:len(*h)-1]
	return x
}
