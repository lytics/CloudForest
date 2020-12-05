package CloudForest

import (
	"bytes"
	"math"
	"testing"
)

// Spot check serialization of strings, float64, and bool
// fields in the forest header.
func TestForestSerializationAndBack(t *testing.T) {
	ftype := Regressor
	targetName := "myTarget"
	l1 := true
	gradboost := 5.2
	predCfg := &PredictConfig{
		Type:       ftype.String(),
		Targetname: targetName,
		L1:         l1,
		Gradboost:  gradboost,
	}

	// write
	var buf bytes.Buffer
	forestwriter := NewForestWriter(&buf)
	icept := 0.5
	forestwriter.WriteForestHeader(0, targetName, icept, predCfg)

	// read back
	forestreader := NewForestReader(&buf)
	forest, err := forestreader.ReadForest()
	if err != nil {
		t.Fatal(err)
	}

	// check
	if !closeEnough(forest.PredCfg.Gradboost, predCfg.Gradboost) {
		t.Errorf("Gradboost did not serialize and restore")
	}
	if !closeEnough(forest.Intercept, icept) {
		t.Errorf("Intercept did not serialize and restore")
	}
	if forest.PredCfg.L1 != predCfg.L1 {
		t.Errorf("L1 did not serialization and restore")
	}
	if forest.Type != ftype {
		t.Errorf("forest.Type did not serialization and restore")
	}
}

const eps = 1e-6

func closeEnough(a, b float64) bool {
	return math.Abs(a-b) < eps
}
