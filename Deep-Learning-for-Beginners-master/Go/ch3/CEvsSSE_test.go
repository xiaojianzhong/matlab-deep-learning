package ch3

import (
	"github.com/xiaojianzhong/matlab-deep-learning/Deep-Learning-for-Beginners-master/Go/utils"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"testing"
)

func TestCEvsSSE(t *testing.T) {
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

	es1 := mat.NewVecDense(1000, nil) // 1000-elem vector
	es2 := mat.NewVecDense(1000, nil) // 1000-elem vector

	wss11 := mat.NewDense(4, 3, utils.Apply(utils.Float64s(4 * 3), func(n float64) float64 {
		return 2 * n - 1
	})) // 4 x 3 matrix
	ws12 := mat.NewVecDense(4, utils.Apply(utils.Float64s(4), func(n float64) float64 {
		return 2 * n - 1
	})) // 4-elem vector
	wss21 := mat.DenseCopyOf(wss11) // 4 x 3 matrix
	ws22 := mat.VecDenseCopyOf(ws12) // 4-elem vector

	// train
	for epoch := 0; epoch < 1000; epoch++ {
		BackpropCE(wss11, ws12, xss, ds)
		BackpropXOR(wss21, ws22, xss, ds)

		e1 := 0.0 // scalar
		e2 := 0.0 // scalar
		n := 4 // scalar
		for k := 0; k < n; k++ {
			xs := xss.RowView(k) // 3-elem vector
			d := ds.AtVec(k) // scalar

			v11 := mat.NewVecDense(4, nil) // 4-elem vector
			v11.MulVec(wss11, xs)
			y11 := mat.NewVecDense(4, nil) // 4-elem vector
			utils.SigmoidVector(y11, v11)
			v12 := mat.Dot(ws12, y11) // scalar
			y12 := utils.SigmoidFloat64(v12) // scalar
			e1 = d - y12 // scalar

			v21 := mat.NewVecDense(4, nil) // 4-elem vector
			v21.MulVec(wss21, xs)
			y21 := mat.NewVecDense(4, nil) // 4-elem vector
			utils.SigmoidVector(y21, v21)
			v22 := mat.Dot(ws22, y21) // scalar
			y22 := utils.SigmoidFloat64(v22) // scalar
			e2 = d - y22 // scalar
		}
		es1.SetVec(epoch, e1 / float64(n))
		es2.SetVec(epoch, e2 / float64(n))
	}

	p, err := plot.New()
	if err != nil {
		t.Error(err)
	}
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Average of Training Error"
	pts1 := make(plotter.XYs, 1000)
	pts2 := make(plotter.XYs, 1000)
	for i := 0; i < 1000; i++ {
		pts1[i].X = float64(i + 1)
		pts1[i].Y = es1.AtVec(i)
		pts2[i].X = float64(i + 1)
		pts2[i].Y = es2.AtVec(i)
	}
	if err = plotutil.AddLines(p, "Cross Entropy", pts1, "Sum of Squared Error", pts2); err != nil {
		t.Error(err)
	}
	if err = p.Save(8 * vg.Inch, 8 * vg.Inch, "result.png"); err != nil {
		t.Error(err)
	}
}

