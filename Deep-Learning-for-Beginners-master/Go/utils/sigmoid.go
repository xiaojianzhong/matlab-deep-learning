package utils

import (
	"math"
)

func SigmoidFloat64(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
