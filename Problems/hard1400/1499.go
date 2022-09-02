package main

import (
	"container/list"
	"fmt"
	"math"
)

/*
yi + yj + |xi - xj| -> (yi - xi) + (yj + xj) --> maximize sum of two pair
xj - xi <= k --> distance of two pair cannot be two large
*/

func findMaxValueOfEquation(points [][]int, k int) int {
	res := -math.MaxInt32
	dq := list.New()
	for _, point := range points {
		fmt.Println(point)
		curX, curY := point[0], point[1]
		// if distance is too large, pop out the front point
		for dq.Len() > 0 {
			frontP := dq.Front()
			frontNode := frontP.Value.(Node1499)
			// fmt.Println("Front", frontNode)
			if curX-frontNode.X > k {
				dq.Remove(frontP)
			} else {
				break
			}
		}
		// update result with cur node
		if dq.Len() > 0 {
			node := dq.Front().Value.(Node1499)
			temp := node.Y - node.X + curY + curX
			fmt.Println(temp)
			if temp > res {
				res = temp
			}
		}
		// if node in dq is smaller than cur pair, pop out back point
		for dq.Len() > 0 {
			backP := dq.Back()
			backNode := backP.Value.(Node1499)
			// fmt.Println("Back", backNode)
			if backNode.Y-backNode.X <= curY-curX {
				dq.Remove(backP)
			} else {
				break
			}
		}
		dq.PushBack(Node1499{curX, curY})
	}
	return res
}

type Node1499 struct {
	X int
	Y int
}

func main() {
	points := [][]int{{1, 3}, {2, 0}, {5, 10}, {6, -10}}
	k := 1
	res := findMaxValueOfEquation(points, k)
	fmt.Println(res)
}
