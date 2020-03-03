package ch5

import (
	"fmt"
	"github.com/azxj/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestDeepDropout(t *testing.T) {
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

	wss1 := mat.NewDense(20, 25, utils.Apply(utils.Float64s(20 * 25), func(n float64) float64 {
		return 2 * n - 1
	})) // 20 x 25 matrix
	wss2 := mat.NewDense(20, 20, utils.Apply(utils.Float64s(20 * 20), func(n float64) float64 {
		return 2 * n - 1
	})) // 20 x 20 matrix
	wss3 := mat.NewDense(20, 20, utils.Apply(utils.Float64s(20 * 20), func(n float64) float64 {
		return 2 * n - 1
	})) // 20 x 20 matrix
	wss4 := mat.NewDense(5, 20, utils.Apply(utils.Float64s(5 * 20), func(n float64) float64 {
		return 2 * n - 1
	})) // 5 x 20 matrix

	// train
	for epoch := 0; epoch < 10000; epoch++ {
		DeepDropout(wss1, wss2, wss3, wss4, xss, dss)
	}

	// inference
	n := 5
	for k := 0; k < n; k++ {
		xs := xss.RowView(k) // 25-elem vector

		vs1 := mat.NewVecDense(20, nil) // 20-elem vector
		vs1.MulVec(wss1, xs)
		ys1 := mat.NewVecDense(20, nil) // 20-elem vector
		utils.SigmoidVector(ys1, vs1)

		vs2 := mat.NewVecDense(20, nil) // 20-elem vector
		vs2.MulVec(wss2, ys1)
		ys2 := mat.NewVecDense(20, nil) // 20-elem vector
		utils.SigmoidVector(ys2, vs2)

		vs3 := mat.NewVecDense(20, nil) // 20-elem vector
		vs3.MulVec(wss3, ys2)
		ys3 := mat.NewVecDense(20, nil) // 20-elem vector
		utils.SigmoidVector(ys3, vs3)

		vs4 := mat.NewVecDense(5, nil) // 5-elem vector
		vs4.MulVec(wss4, ys3)
		ys4 := mat.NewVecDense(5, nil) // 5-elem vector
		utils.Softmax(ys4, vs4)

		fmt.Println(ys4)
	}
}
