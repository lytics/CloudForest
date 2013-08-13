Package CloudForest implements ensembles of decision trees for machine
learning in pure Go (golang). It includes implementations of Breiman
and Cutler's Random Forest for classification and regression on heterogeneous
numerical/categorical data with missing values and several related algorithms
including entropy and cost driven classification, L1 regression and feature
selection with artificial contrasts and hooks for modifying the algorithms
for your needs.

Command line utilities to grow, apply and canalize forests are provided in sub
directories.

CloudForest is being developed in the Shumelivich Lab at the Institute for Systems
Biology.


Documentation has been generated with godoc and can be viewed live at:
http://godoc.org/github.com/ryanbressler/CloudForest

Pull requests and bug reports are welcome; Code Repo and Issue tracker can be found at:
https://github.com/ryanbressler/CloudForest


Goals

CloudForest is intended to provide fast, comprehensible building blocks that can
be used to implement ensembles of decision trees. CloudForest is written in Go to
allow a data scientist to develop and scale new models and analysis quickly instead
of having to modify complex legacy code.

Data structures and file formats are chosen with use in multi threaded and cluster 
environments in mind.


    