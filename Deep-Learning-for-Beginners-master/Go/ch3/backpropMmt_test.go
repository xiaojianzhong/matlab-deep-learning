package ch3

import (
	"fmt"
	"github.com/azxj/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestBackpropMmt(t *testing.T) {
	xss := mat.NewDense(4, 3, []float64{
		0, 0, 1,
		0, 1, 1,
		1, 0, 1,
		1, 1, 1,
	}) // 4 x 3 matrix
	ds := mat.NewVecDense(4, []float64{
		0,
		1,
		1,
		0,
	}) // 4-elem vector

	wss1 := mat.NewDense(4, 3, utils.Apply(utils.Float64s(4 * 3), func(n float64) float64 {
		return 2 * n - 1
	})) // 4 x 3 matrix
	ws2 := mat.NewVecDense(4, utils.Apply(utils.Float64s(4), func(n float64) float64 {
		return 2 * n - 1
	})) // 4-elem vector

	// train
	for epoch := 0; epoch < 10000; epoch++ {
		BackpropMmt(wss1, ws2, xss, ds)
	}

	// inference
	n := 4 // scalar
	for k := 0; k < n; k++ {
		xs := xss.RowView(k) // 3-elem vector
		v1 := mat.NewVecDense(4, nil) // 4-elem vector
		v1.MulVec(wss1, xs)
		y1 := mat.NewVecDense(4, nil) // 4-elem vector
		utils.SigmoidVector(y1, v1)
		v2 := mat.Dot(ws2, y1) // scalar
		y2 := utils.SigmoidFloat64(v2) // scalar
		fmt.Println(y2)
	}
}
