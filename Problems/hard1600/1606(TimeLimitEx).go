package main

import "fmt"

func busiestServers(k int, arrival []int, load []int) []int {
	endTime := map[int]int{}
	handledReqs := map[int]int{}
	// iterate each request
	for idx, start := range arrival {
		end := start + load[idx]
		fmt.Printf("R%v has range %v, %v\n", idx+1, start, end)
		startServer := idx % k
		// iterate each server starting from i % k server
		for i := 0; i < k; i++ {
			curServer := (startServer + i) % k
			// if current server is busy, try next server
			if _, ok := endTime[curServer]; ok && endTime[curServer] > start {
				continue
			}
			endTime[curServer] = end
			handledReqs[curServer] += 1
			break
		}
		fmt.Println("endTime", endTime)
		fmt.Println("handledReqs", handledReqs)
	}
	return GetServersWithMaxReq(handledReqs)
}

func GetServersWithMaxReq(handledReqs map[int]int) []int {
	keys := []int{}
	maxReqNum := 0
	for server, reqNum := range handledReqs {
		if reqNum > maxReqNum {
			keys = []int{}
			keys = append(keys, server)
			maxReqNum = reqNum
		} else if reqNum == maxReqNum {
			keys = append(keys, server)
		}
	}
	return keys
}

func main() {
	k := 3
	arrival := []int{1, 2, 3, 4, 5}
	load := []int{5, 2, 3, 3, 3}
	res := busiestServers(k, arrival, load)
	fmt.Println(res)
}
