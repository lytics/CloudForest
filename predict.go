package CloudForest

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Predictions is returned by Predict().
// Only one of Real or Cat will be populated.
type Predictions struct {
	// Real will be populated for regression predictions.
	Real []float64

	// Cat (category) will be populated for classifiers.
	Cat []string

	// IsReal will be true when Real is populated. Otherwise
	// Cat (above) is in use.
	IsReal bool
}

// helper for Predict()
func getTallyerForForest(fm *FeatureMatrix, f *Forest) VoteTallyer {
	n := fm.Data[fm.Map[f.Target]].Length()

	if f.PredCfg.Gradboost != 0 || f.PredCfg.Adaboost {
		return NewSumBallotBox(n)
	}

	switch f.Type {
	case Regressor:
		return NewNumBallotBox(n)
	case Classifier:
		return NewCatBallotBox(n)
	}
	panic(fmt.Sprintf("which BallotBox to use for ForestType %v ?", f.Type))
}

// Predict is ForestType agnostic, generating preditions
// with whichever ForestType (Classifier, Regressor, GBM)
// is provided in f. Argument fm must have features comparable to
// the FeatureMatrix that f was trained on, but fm may, and
// frequently will, contain previously unseen data.
//
func Predict(fm *FeatureMatrix, f *Forest) (*Predictions, error) {
	tallyer := getTallyerForForest(fm, f)
	for _, tree := range f.Trees {
		tree.Vote(fm, tallyer)
	}

	N := fm.Data[0].Length()

	switch box := tallyer.(type) {
	case *CatBallotBox:
		var preds []string
		for i := 0; i < N; i++ {
			preds = append(preds, box.Tally(i))
		}
		return &Predictions{Cat: preds}, nil

	case *NumBallotBox:
		var preds []float64
		for i := 0; i < N; i++ {
			pred := box.TallyNum(i) + f.Intercept
			preds = append(preds, pred)
		}
		return &Predictions{Real: preds, IsReal: true}, nil

	case *SumBallotBox:
		var preds []float64
		for i := 0; i < N; i++ {
			pred := box.TallyNum(i) + f.Intercept
			if f.PredCfg.Expit {
				pred = Expit(pred)
			}
			preds = append(preds, pred)
		}
		return &Predictions{Real: preds, IsReal: true}, nil
	default:
		return nil, fmt.Errorf("Unknown tallyer type %T in Predict()", tallyer)
	}
}

// read back predictions written to file
// by applyforest

type RealPredFile struct {
	Rowname    []string
	Prediction []float64
	Actual     []float64
}

func ReadTabSeparatedPredictions(fn string) (rpf *RealPredFile, err error) {

	var f *os.File
	f, err = os.Open(fn)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)

	rpf = &RealPredFile{}

	for scanner.Scan() {
		line := scanner.Text()
		fld := strings.Split(line, "\t")
		for j := range fld {
			fld[j] = strings.TrimSpace(fld[j])
		}

		rpf.Rowname = append(rpf.Rowname, fld[0])

		f64, err := strconv.ParseFloat(fld[1], 64)
		if err != nil {
			f64 = nan
		}
		rpf.Prediction = append(rpf.Prediction, f64)

		f64, err = strconv.ParseFloat(fld[2], 64)
		if err != nil {
			f64 = nan
		}
		rpf.Actual = append(rpf.Actual, f64)
	}
	err = scanner.Err()
	return
}
