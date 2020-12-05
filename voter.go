package CloudForest

// VoteTallyer  is used to tabulate votes by
// trees and is implemented by feature type specific
// structs like NumBallotBox and CatBallotBox.
type VoteTallyer interface {

	// Vote should register a vote that casei should be predicted as pred.
	Vote(casei int, pred string, weight float64)

	// TallyError returns the error vs the supplied feature.
	TallyError(feature Feature) float64

	// Tally returns the predicted category string for casei.
	Tally(casei int) string

	// TallyNum returns the predicted regression for case i.
	// Classifiers should just return NaN.
	TallyNum(i int) (predicted float64)
}
