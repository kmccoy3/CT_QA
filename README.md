# Vascular Object Identification Tool

This repository provides source code and figure generation for K. McCoy et al. (2024) 

## Table of Contents

* [Technologies](#technologies)
* [Usage](#usage)
* [Project Status](#project-status)
* [Credits](#credits)
* [License](#license)

## Technologies

The scripts and notebooks contained in this repo use the following packages:

* Python == 3.9.6
  * numpy == 1.21.5
  * pandas == 1.4.2
  * random
  * math
  * scikit-learn == 1.0.2
  * collections
  * matplotlib == 3.5.1
  * xgboost == 1.7.3
  * seaborn == 0.11.2

## Usage

#### Useful Links:
* Paper: [LINK](https://aapm.onlinelibrary.wiley.com/journal/24734209)
* Public Image Dataset: [LINK](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)

#### Explanation of each file:
* eda_figures.ipynb
  * Calculate summary statistics and produce exploratory data analysis (EDA) plots.
* method_comparison.py
  * Conduct 5-fold cross validation on your choice of ML model.
* gridsearch_CV.py
  * Conduct grid search cross validation on your choice of ML model with specified hyperparameter choices.
* test.py
  * Train chosen model on all train-validation data and test on holdout test data set.
* intensity_validation.ipynb
  * Calculate intensity values after trained model on holdout test data.
* external_validation.ipynb
  * Calculates intensity values and performance on external validation data.

## Project Status

This project is currently released for public use and is being actively maintained.

## Credits

To cite this project and/or code, please use `citation.bib`. 

Alternatively, use GitHub's built in citation feature / `CITATION.cff`.

## License

This project uses the MIT license. In addition to the terms of this license, any work using this code must cite this repository using the instructions above.