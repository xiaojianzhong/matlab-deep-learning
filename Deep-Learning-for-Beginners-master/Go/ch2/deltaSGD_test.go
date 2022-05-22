package ch2

import (
	"fmt"
	"github.com/xiaojianzhong/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestDeltaSGD(t *testing.T) {
	xss := mat.NewDense(4, 3, []float64{
		0, 0, 1,
		0, 1, 1,
		1, 0, 1,
		1, 1, 1,
	}) // 4 x 3 matrix

	ds := mat.NewVecDense(4, []float64{
		0,
		0,
		1,
		1,
	}) // 4-elem vector

	ws := mat.NewVecDense(3, utils.Apply(utils.Float64s(3), func(n float64) float64 {
		return 2 * n - 1
	})) // 3-elem vector

	// train
	for epoch := 0; epoch < 10000; epoch++ {
		DeltaSGD(ws, xss, ds)
	}

	// inference
	n := 4 // scalar
	for k := 0; k < n; k++ {
		xs := xss.RowView(k) // 3-elem vector
		v := mat.Dot(ws, xs) // scalar
		y := utils.SigmoidFloat64(v) // scalar
		fmt.Println(y)
	}
}
