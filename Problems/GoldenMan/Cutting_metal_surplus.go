package main

import "fmt"

/* #
# HackerRank - Cutting Metal Surplus
#
# The owner of a metal rod factory has a surplus of rods of arbitrary lengths. A local contractor offers to buy any of the factory's
# surplus as long as all the rods have the same exact integer length, referred to as saleLength. The factory owner can increase the
# number of sellable rods by cutting each rod zero or more times, but each cut has a cost denoted by costPerCut. After all cuts have
# been made, any leftover rods having a length other than saleLength must be discarded for no profit. The factory owner's total profit
# for the sale is calculated as:
#
#      totalProfit = totalUniformRods × saleLength × salePrice − totalCuts × costPerCut
#
# where totalUniformRods is the number of sellable rods, salePrice is the per unit length price that the contractor agrees to pay, and
# totalCuts is the total number of times the rods needed to be cut.
#
#
# Complete the function maxProfit. The function must return an integer that denotes the maximum possible profit.
#
# maxProfit has the following parameter(s):
#    costPerCut:  integer cost to make a cut
#    salePrice:  integer per unit length sales price
#    lengths[lengths[0],...lengths[n-1]]:  an array of integer rod lengths
#
# Constraints
#
# 1 ≤ n ≤ 50
# 1 ≤ lengths[i] ≤ 104
# 1 ≤ salePrice, costPerCut ≤ 1000 */

func CutMetal(costPerCut int, salePrices int, lengths []int) int {
	res := 0
	maxLength := GetMaxLength(lengths)
	for saleLength := 1; saleLength <= maxLength; saleLength++ {
		profit := 0
		totalCost := 0
		for _, length := range lengths {
			cuts := length/saleLength - 1
			totalCost += cuts * costPerCut
			numSaleLength := length / saleLength
			curProfit := numSaleLength*saleLength*salePrices - totalCost
			if curProfit > 0 {
				profit += curProfit
			}
		}
		if profit > res {
			res = profit
		}
	}
	return res
}

func GetMaxLength(lengths []int) int {
	res := 0
	for _, length := range lengths {
		if length > res {
			res = length
		}
	}
	return res
}

func main() {
	costPercut := 2
	salePrices := 2
	lengths := []int{5, 10, 15}
	res := CutMetal(costPercut, salePrices, lengths)
	fmt.Println(res)
}
