package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/lytics/CloudForest"
)

func main() {
	fm := flag.String("fm",
		"featurematrix.afm", "AFM formated feature matrix containing data.")
	rf := flag.String("rfpred",
		"rface.sf", "A predictor forest.")
	predfn := flag.String("preds",
		"", "The name of a file to write the predictions into.")

	flag.Parse()

	//Parse Data
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

	forestfile, err := os.Open(*rf) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestreader := CloudForest.NewForestReader(forestfile)
	forest, err := forestreader.ReadForest()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Forest is of type %v.\n", forest.Type.String())

	var predfile *os.File
	if *predfn != "" {
		predfile, err = os.Create(*predfn)
		if err != nil {
			log.Fatal(err)
		}
		defer predfile.Close()
	}
	targeti, hasTarget := data.Map[forest.Target]

	preds, err := CloudForest.Predict(data, forest)
	if err != nil {
		log.Fatal(err)
	}
	if *predfn != "" {
		fmt.Printf("Outputting label predicted actual tsv to %v\n", *predfn)
		for i, l := range data.CaseLabels {
			actual := "NA"
			if hasTarget {
				actual = data.Data[targeti].GetStr(i)
			}

			var result string
			if preds.IsReal {
				result = fmt.Sprintf("%v", preds.Real[i])
			} else {
				result = preds.Cat[i]
			}

			fmt.Fprintf(predfile, "%v\t%v\t%v\n", l, result, actual)
		}
	}

}
