package main

import "fmt"

/*
	Union Find
	1.	Abandon all edges of type3 and get common roots
	2.	Abandon all edges of type1 / Abandon all edges of type2 with common roots from step1

	Reachable or not
	if e1 == n - 1 && e2 == n - 1, return res else return -1
*/
func maxNumEdgesToRemove(n int, edges [][]int) int {
	res := 0
	edges1, edges2, edges3 := [][]int{}, [][]int{}, [][]int{}
	for _, edge := range edges {
		edgeType, src, dest := edge[0], edge[1], edge[2]
		if edgeType == 1 {
			edges1 = append(edges1, []int{src, dest})
		} else if edgeType == 2 {
			edges2 = append(edges2, []int{src, dest})
		} else {
			edges3 = append(edges3, []int{src, dest})
		}
	}
	// fmt.Println(edges1, edges2, edges3)
	roots := GetRoots(n)
	// fmt.Println(roots)
	// iterate edges3
	e1, e2 := 0, 0
	for _, edge := range edges3 {
		ConnectCommon(roots, edge[0], edge[1], &res, &e1, &e2)
	}
	// fmt.Println(roots, res)
	// iterate edges1 / edges2 with the same root
	roots1, roots2 := CopyRoots(roots), CopyRoots(roots)
	for _, edge := range edges1 {
		Connect(roots1, edge[0], edge[1], &res, &e1)
	}
	// fmt.Println(roots1, res)
	for _, edge := range edges2 {
		Connect(roots2, edge[0], edge[1], &res, &e2)
	}
	// fmt.Println(roots2, res)
	if e1 == n-1 && e2 == n-1 {
		return res
	}
	return -1
}

func GetRoots(n int) []int {
	roots := make([]int, n+1)
	for i := range roots {
		roots[i] = i
	}
	return roots
}

func Find(roots []int, i int) int {
	if roots[i] != i {
		// fmt.Println(i, roots[i])
		roots[i] = Find(roots, roots[i])
	}
	return roots[i]
}

func ConnectCommon(roots []int, i, j int, res *int, e1 *int, e2 *int) {
	r1, r2 := Find(roots, i), Find(roots, j)
	// fmt.Println(i, j, "root", r1, r2)
	if r1 != r2 {
		roots[r1] = r2
		*e1 += 1
		*e2 += 1
	} else {
		onnect(roots []int, i, j int, res *int, e *int) {
	r1, r2 := Find(roots, i), Find(roots, j)
	// fmt.Println(i, j, "root", r1, r2)
	if r1 != r2 {
		roots[r1] = r2
		*e += 1
	} else {
		*res += 1
	}
}

func CopyRoots(roots []int) []int {
	n := len(roots)
	newRoots := make([]int, n)
	for i := range roots {
		newRoots[i] = roots[i]
	}
	return newRoots
}*res += 1
	}
}

func C

func main() {
	n := 4
	egdes := [][]int{{3, 1, 2}, {3, 2, 3}, {1, 1, 3}, {1, 2, 4}, {1, 1, 2}, {2, 3, 4}}
	res := maxNumEdgesToRemove(n, egdes)
	fmt.Println(res)
}
