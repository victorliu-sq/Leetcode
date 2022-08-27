func reachingPoints(sx int, sy int, tx int, ty int) bool {
	for {
		// check if reach target
		if sx == tx && sy == ty {
			return true
		}
		// if target cannot shrink anymore
		if tx == ty || tx < sx || ty < sy {
			return false
		}
		// check if only one side reaches target
		if sx == tx {
			return (ty-sy)%sx == 0
		} else if sy == ty {
			return (tx-sx)%sy == 0
		}
		// shrink the bigger side
		if tx > ty {
			tx = tx % ty
		} else {
			ty = ty % tx
		}
	}
}