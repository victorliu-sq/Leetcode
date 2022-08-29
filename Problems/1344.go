package main

import "fmt"

func angleClock(hour int, minutes int) float64 {
	degreePerHour := 360.0 / 12
	degreePerMinute := 360.0 / 60
	// fmt.Println(degreePerHour, degreePerMinute)
	degreeMinute := float64(minutes) * degreePerMinute
	degreeHour := float64(hour%12)*degreePerHour + float64(minutes)*degreePerHour/60 // minutes / 60 * degreePerHour
	fmt.Println(degreeHour, degreeMinute)
	return GetInterDegree(degreeHour, degreeMinute)
}

func GetInterDegree(degreeHour float64, degreeMinute float64) float64 {
	// Get the degree between two pointers
	var interDegree float64
	if degreeHour > degreeMinute {
		interDegree = degreeHour - degreeMinute
	} else {
		interDegree = degreeMinute - degreeHour
	}
	// if the degree > 180, we need to get degree to complement it
	if interDegree > 180 {
		return 360 - interDegree
	}
	return interDegree
}

func main() {
	hour := 12
	minutes := 30
	res := angleClock(hour, minutes)
	fmt.Println(res)
}
