package main

import (
	"fmt"
	"math"
)

/*
DFS: 	dfs(cur) = min( 1 + dfs(next) ) , dfs return Term for base case --> wrong
		dfs(cur) = min( dfs(next) ) dfs return 0 for base case --> correct
*/

func minNumberOfSemesters(n int, relations [][]int, k int) int {
	degree := GetDegree(relations, n)
	fmt.Println(degree)
	nextCourses := GetNextCourse(relations)
	fmt.Println(nextCourses)
	dp := map[int]int{}
	res := dfs1494(0, n, 0, k, degree, nextCourses, dp)
	return res
}

func dfs1494(curCourseNum, n, bitmask, k int, degree []int, nextCourses map[int][]int, dp map[int]int) int {
	// 1. Base Case
	// fmt.Println(curCourseNum, strconv.FormatInt(int64(bitmask), 2), degree)
	if curCourseNum == n {
		return 0
	}
	// 2. Check DP
	if _, ok := dp[bitmask]; ok {
		return dp[bitmask]
	}
	// 3. Next Recursion
	// Get all courses that we can take now
	canTake := []int{}
	for course := 1; course <= n; course++ {
		if degree[course] == 0 && !checkCourseTaken(bitmask, course) {
			canTake = append(canTake, course)
		}
	}
	// fmt.Println(canTake)
	res := math.MaxInt32
	// Generate all possible combinations of length k
	combinations := GetCombinations(canTake, min1494(len(canTake), k))
	// fmt.Println(combinations)
	for _, courses := range combinations {
		// fmt.Println(courses)
		newBitmask := bitmask
		newDegree := CopyDegree(degree)
		for _, course := range courses {
			// fmt.Println("nextCourse:", course)
			// take course = (1) newbitmask (2) degree of next course -= 1
			newBitmask |= (1 << course)
			for _, nextCourse := range nextCourses[course] {
				newDegree[nextCourse] -= 1
			}
		}
		// fmt.Println(newDegree, strconv.FormatInt(int64(newBitmask), 2))
		newCourseNum := curCourseNum + min1494(len(canTake), k)
		temp := 1 + dfs1494(newCourseNum, n, newBitmask, k, newDegree, nextCourses, dp)
		if temp < res {
			res = temp
		}
	}
	// 4. Update dp
	dp[bitmask] = res
	return res
}

func GetDegree(relations [][]int, n int) []int {
	degree := make([]int, n+1)
	for _, relation := range relations {
		post := relation[1]
		degree[post] += 1
	}
	return degree
}

func GetNextCourse(relations [][]int) map[int][]int {
	nextCourses := make(map[int][]int)
	for _, relation := range relations {
		prev, post := relation[0], relation[1]
		if _, ok := nextCourses[prev]; !ok {
			nextCourses[prev] = []int{}
		}
		nextCourses[prev] = append(nextCourses[prev], post)
	}
	return nextCourses
}

func min1494(n1, n2 int) int {
	if n1 < n2 {
		return n1
	}
	return n2
}

func checkCourseTaken(bitmask, course int) bool {
	if bitmask&(1<<course) != 0 {
		return true
	}
	return false
}

// ***********************************************************************
func GetCombinations(nums []int, k int) [][]int {
	combinations := [][]int{}
	dfsCombination(0, k, []int{}, nums, &combinations)
	return combinations
}

func dfsCombination(start, k int, curNums, nums []int, combiantions *[][]int) {
	if len(curNums) == k {
		// fmt.Println(curNums)
		*combiantions = append(*combiantions, curNums)
		return
	}
	// not enough nums
	n := len(nums)
	if len(curNums)+n-1-start+1 < k {
		return
	}
	end := n - 1
	for i := start; i <= end; i++ {
		newNums := CopySlice1494(curNums)
		newNums = append(newNums, nums[i])
		// fmt.Println(newNums)
		dfsCombination(i+1, k, newNums, nums, combiantions)
	}
}

func CopyDegree(degree []int) []int {
	n := len(degree)
	newDegree := make([]int, n)
	for i := range degree {
		newDegree[i] = degree[i]
	}
	return newDegree
}

func CopySlice1494(arr []int) []int {
	res := []int{}
	for _, num := range arr {
		res = append(res, num)
	}
	return res
}

func main() {
	// n := 4
	// relations := [][]int{{2, 1}, {3, 1}, {1, 4}}
	// k := 2
	n := 5
	relations := [][]int{{2, 1}, {3, 1}, {4, 1}, {1, 5}}
	k := 2
	res := minNumberOfSemesters(n, relations, k)
	fmt.Println(res)
}
