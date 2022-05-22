package ch4

import (
	"fmt"
	"github.com/xiaojianzhong/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestMultiClass(t *testing.T) {
	Rng(3)

	xss := mat.NewDense(5, 25, []float64{
		0, 1, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 1, 1, 1, 0,

		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		1, 0, 0, 0, 0,
		1, 1, 1, 1, 1,

		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 1, 1, 0,

		0, 0, 0, 1, 0,
		0, 0, 1, 1, 0,
		0, 1, 0, 1, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 1, 0,

		1, 1, 1, 1, 1,
		1, 0, 0, 0, 0,
		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 1, 1, 0,
	}) // 5 x 25 matrix
	dss := mat.NewDense(5, 5, []float64{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1,
	}) // 5 x 5 matrix

	wss1 := mat.NewDense(50, 25, utils.Apply(utils.Float64s(50 * 25), func(n float64) float64 {
		return 2 * n - 1
	})) // 50 x 25 matrix
	wss2 := mat.NewDense(5, 50, utils.Apply(utils.Float64s(5 * 50), func(n float64) float64 {
		return 2 * n - 1
	})) // 5 x 50 matrix

	// train
	for epoch := 0; epoch < 10000; epoch++ {
		MultiClass(wss1, wss2, xss, dss)
	}

	// inference
	n := 5
	for k := 0; k < n; k++ {
		xs := xss.RowView(k) // 25-elem vector
		vs1 := mat.NewVecDense(50, nil) // 50-elem vector
		vs1.MulVec(wss1, xs)
		ys1 := mat.NewVecDense(50, nil) // 50-elem vector
		utils.SigmoidVector(ys1, vs1)
		vs2 := mat.NewVecDense(5, nil) // 5-elem vector
		vs2.MulVec(wss2, ys1)
		ys2 := mat.NewVecDense(5, nil) // 5-elem vector
		utils.Softmax(ys2, vs2)
		fmt.Println(ys2)
	}

	realXss := mat.NewDense(5, 25, []float64{
		0, 0, 1, 1, 0,
		0, 0, 1, 1, 0,
		0, 1, 0, 1, 0,
		0, 0, 0, 1, 0,
		0, 1, 1, 1, 0,

		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		1, 0, 0, 0, 1,
		1, 1, 1, 1, 1,

		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		1, 0, 0, 0, 1,
		1, 1, 1, 1, 0,

		0, 1, 1, 1, 0,
		0, 1, 0, 0, 0,
		0, 1, 1, 1, 0,
		0, 0, 0, 1, 0,
		0, 1, 1, 1, 0,

		0, 1, 1, 1, 1,
		0, 1, 0, 0, 0,
		0, 1, 1, 1, 0,
		0, 0, 0, 1, 0,
		1, 1, 1, 1, 0,
	}) // 5 x 25 matrix
	for k := 0; k < n; k++ {
		xs := realXss.RowView(k) // 25-elem vector
		vs1 := mat.NewVecDense(50, nil) // 50-elem vector
		vs1.MulVec(wss1, xs)
		ys1 := mat.NewVecDense(50, nil) // 50-elem vector
		utils.SigmoidVector(ys1, vs1)
		vs2 := mat.NewVecDense(5, nil) // 5-elem vector
		vs2.MulVec(wss2, ys1)
		ys2 := mat.NewVecDense(5, nil) // 5-elem vector
		utils.Softmax(ys2, vs2)
		fmt.Println(ys2)
	}
}
