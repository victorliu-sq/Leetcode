package main

import "fmt"

func LotteryCoupons(N int) int {
	coupons := []int{}
	for i := 1; i <= N; i++ {
		coupons = append(coupons, i)
	}
	sums := []int{}
	for _, coupon := range coupons {
		sums = append(sums, GetSumOfEachNum(coupon))
	}
	// fmt.Println(sums)
	sumNum := make(map[int]int)
	for _, sum := range sums {
		sumNum[sum] += 1
	}
	fmt.Println(sumNum)
	res := 0
	maxSum := 0
	countNum := make(map[int]int)
	for sum, sumCount := range sumNum {
		// fmt.Println(count)
		if sum >= maxSum {
			maxSum = sum
			countNum[sumCount] += 1
			if countNum[sumCount] > res {
				res = countNum[sumCount]
			}
		}
	}
	return res
}

func GetSumOfEachNum(coupon int) int {
	sum := 0
	for coupon > 0 {
		digit := coupon % 10
		sum += digit
		coupon /= 10
	}
	return sum
}

func main() {
	N := 12
	res := LotteryCoupons(N)
	fmt.Println(res)
}
