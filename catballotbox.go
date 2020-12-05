package CloudForest

import (
	"sort"
	"sync"
)

//CatBallot is used insideof CatBallotBox to record catagorical votes in a thread safe
//manner.
type CatBallot struct {
	Mutex sync.RWMutex
	Map   map[int]float64
}

//NewCatBallot returns a pointer to an initalized CatBallot with a 0 size Map.
func NewCatBallot() (cb *CatBallot) {
	cb = new(CatBallot)
	cb.Map = make(map[int]float64, 0)
	return
}

//CatBallotBox keeps track of votes by trees in a thread safe manner.
type CatBallotBox struct {
	*CatMap
	Box []*CatBallot
}

//NewCatBallotBox builds a new ballot box for the number of cases specified by "size".
func NewCatBallotBox(size int) *CatBallotBox {
	bb := CatBallotBox{
		CatMap: NewCatMap(),
		Box:    make([]*CatBallot, 0, size),
	}
	for i := 0; i < size; i++ {
		bb.Box = append(bb.Box, NewCatBallot())
	}
	return &bb
}

//Vote registers a vote that case "casei" should be predicted to be the
//category "pred".
func (bb *CatBallotBox) Vote(casei int, pred string, weight float64) {
	predn := bb.CatToNum(pred)
	bb.Box[casei].Mutex.Lock()
	if _, ok := bb.Box[casei].Map[predn]; !ok {
		bb.Box[casei].Map[predn] = 0
	}
	bb.Box[casei].Map[predn] = bb.Box[casei].Map[predn] + weight
	bb.Box[casei].Mutex.Unlock()
}

//Tally tallies the votes for the case specified by i as
//if it is a Categorical or boolean feature. Ie it returns the mode
//(the most frequent value) of all votes.
func (bb *CatBallotBox) Tally(i int) (predicted string) {
	var predictedn int
	var maxVote float64
	var ties []int
	bb.Box[i].Mutex.RLock()
	for k, v := range bb.Box[i].Map {
		if v > maxVote {
			predictedn = k
			maxVote = v
			ties = nil
		}

		// keep track of the ties so that our predictions
		// are deterministic
		if v == maxVote {
			ties = append(ties, k)
		}
	}
	bb.Box[i].Mutex.RUnlock()

	// if there is a tie in the predictions,
	// then pick the smaller key
	if len(ties) > 1 {
		sort.Ints(ties)
		predictedn = ties[0]
	}

	if maxVote > 0 {
		predicted = bb.Back[predictedn]
	} else {
		predicted = "NA"
	}
	return
}

/*
TallyError returns the balanced classification error for categorical features.

1 - sum((sum(Y(xi)=Y'(xi))/|xi|))

where
Y are the labels
Y' are the estimated labels
xi is the set of samples with the ith actual label

Case for which the true category is not known are ignored.

*/
func (bb *CatBallotBox) TallyError(feature Feature) float64 {
	catfeature := feature.(CatFeature)
	ncats := catfeature.NCats()
	correct := make([]int, ncats)
	total := make([]int, ncats)
	for i := 0; i < feature.Length(); i++ {
		value := catfeature.Geti(i)
		predicted := bb.Tally(i)
		if feature.IsMissing(i) {
			continue
		}
		total[value]++
		if catfeature.NumToCat(value) == predicted {
			correct[value]++
		}
	}

	var e float64
	for i, ncorrect := range correct {
		e += float64(ncorrect) / float64(total[i])
	}

	e /= float64(ncats)
	e = 1.0 - e
	return e
}

// TallyNum always returns NaN for a SumBallotBox; we
// meet the demands of the expanded VoteTallyer interface.
func (bb *CatBallotBox) TallyNum(i int) (predicted float64) {
	return nan
}
