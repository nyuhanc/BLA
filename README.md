Task: Find the best linear approximation (BLA) for the given datasets: PRBS input defines the training set and the trian of impulses defines the test set.
Solution: - 0_preliminary_analysis.ipynb: consist the preliminary data analysis
          - 1_ARX_FROLS.ipynb: the ARX model implementation based on the systidentpy lib. (works well, but no option to include MA part)
          - 2_IV.ipynb: the instrumental variables method (hand written implementation. doesn't work very well')
          - 4_ARX_FROLS.ipynb same as (2) but with overparametrization and exhaustive search over model orders to find the best model

Comments: the linearmodels and (its wrapper) pymarima libraries that should be most suitable for the given task are utter trash
