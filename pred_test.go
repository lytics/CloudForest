package CloudForest

import (
	"fmt"
	"log"
	"os"
	"testing"
)

// Check the Predict() functionality, comparing
// to output from earlier applyforest code. We
// use a model from an already serialized
// test forest, saved in the data/ dir.
func TestPredict(t *testing.T) {

	model := "data/pred_test_forestfire_model.sf"
	fd, err := os.Open(model)
	if err != nil {
		t.Fatal(err)
	}
	defer fd.Close()
	forestreader := NewForestReader(fd)
	forest, err := forestreader.ReadForest()
	if err != nil {
		t.Fatal(err)
	}

	// data to predict
	forestFireDataFn := "data/forestfires.trans.fm"
	data, err := LoadAFM(forestFireDataFn)
	if err != nil {
		log.Fatal(err)
	}

	// expected predictions
	expectedFilename := "data/pred_test_expected_predictions.tsv"
	expected, err := ReadTabSeparatedPredictions(expectedFilename)
	if err != nil {
		t.Fatal(err)
	}

	preds, err := Predict(data, forest)
	if err != nil {
		t.Fatal(err)
	}
	if !preds.IsReal {
		t.Fatalf("expected Real predictions from forest fire regression")
	}
	N := data.Data[0].Length()
	for i := 0; i < N; i++ {
		if !closeEnough(preds.Real[i], expected.Prediction[i]) {
			t.Fatalf("i=%v was off: preds.Real[i]=%v was not close to expected.Prediction[i]=%v", i, preds.Real[i], expected.Prediction[i])
		}
	}
	fmt.Printf("all %v rows matched expected\n", N)
}
