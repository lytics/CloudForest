package CloudForest

// PredictConfig holds options that are set
// during growforest and that are needed during applyforest
// to correctly predict from the forest.
//
type PredictConfig struct {

	// Type is "Classifier" or "Regressor"
	Type string

	// Targetname gives the row header of the target in the feature matrix.
	Targetname string

	// Costs is for categorical targets: a json string to float map of the cost of falsely identifying each category.
	Costs string

	// Dentropy gives the Class disutilities for disutility entropy.
	Dentropy string

	// Adacosts gives the Json costs for cost sentive AdaBoost.
	Adacosts string

	// Rfweights gives, for categorical targets, a json string to float map of the weights to use for each category in Weighted RF.
	Rfweights string

	// Blacklist is a list of feature id's to exclude from the set of predictors.
	Blacklist string

	// L1 means use l1 norm regression (target must be numeric)
	L1 bool

	// Density means build density estimating trees instead of classification/regression trees
	Density bool

	// Positive class to output probabilities for.
	Positive string

	// Entropy means use entropy minimizing classification (target must be categorical).
	Entropy bool

	// Adaptive boosting for regression/classification.
	Adaboost bool

	// Gradboost means do gradient boosting with the specified learning rate.
	Gradboost float64

	// Ordinal means use ordinal regression (target must be numeric).
	Ordinal bool

	// Expit means transform with expit (inverse logit): for gradient boosting classification.
	Expit bool
}
