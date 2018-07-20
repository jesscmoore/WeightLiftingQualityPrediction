# Final Practical Machine Learning Assignment: Predicting Weight Lifting Quality

## Contents
This repository contains the project report in html and md format, the accompanying Rmd script, 1 R scripts to download the data, and 4 png chart files. For full reproducibility, the data files are included also.
- ``predictLiftingQuality.html`` - project report showing code output and figures in html
- ``predictLiftingQuality.md`` - project report in md format
- ``predictLiftingQuality.Rmd`` - project report R markdown file
- ``downloadData.R`` - code extract that downloads the activity data only
- ``pml-training.csv`` - the training data 
- ``pml-testing.csv`` - the test data
- ``predictLiftingQuality_files/figure-html/modelacc-1.png`` - plot of the K-fold cross validation accuracy
- ``predictLiftingQuality_files/figure-html/modelacc-2.png`` - plot of the error as a function of number of random forest trees
- ``predictLiftingQuality_files/figure-html/modelacc-3.png`` - plot of the 10 variables with highest predictor importance
- ``predictLiftingQuality_files/figure-html/validate-1.png`` - plot of the match of the prediction to the actual class on the validation set



## Analysis

The Rmd ``predictLiftingQuality.Rmd`` script downloads the data, builds and applies a predictive model, and creates a html report of the analysis. 


## How to use *.R

Run knitr on ```predictLiftingQuality.Rmd``` to download the data, perform the analysis and produce the report. The report describes the problem, data, analysis method and results.

Downloaded data files:
- ``pml-training.csv`` - the training data
- ``pml-testing.csv`` - the test data



### Dependencies
```predictLiftingQuality.Rmd``` depends on ```dplyr``` and ```gpplot2```. If not installed, it will download and install the these packages.

