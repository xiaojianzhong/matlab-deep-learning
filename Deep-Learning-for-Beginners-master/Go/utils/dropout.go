package utils

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

func Dropout(recv *mat.VecDense, ratio float64) {
	size := recv.Len()
	numSurvived := int(math.Round(float64(size) * (1.0 - ratio)))
	indexes := rand.Perm(size)
	zeros := mat.NewVecDense(size, nil)
	for i := 0; i < numSurvived; i++ {
		zeros.SetVec(indexes[i], 1.0 / (1.0 - ratio))
	}
	recv.MulElemVec(recv, zeros)
}
