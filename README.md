# Unfolding Beijing in a Hedonic Way

* [Wei Lin](http://www.weilinmetrics.com/), [Zhentao Shi](http://www.zhentaoshi.com/), Yishu Wang and Ting Hin Yan: [“Unfolding Beijing in a Hedonic Way”](https://www.researchgate.net/publication/339551353_Unfolding_Beijing_in_a_Hedonic_Way).

This repository contains the data and R scripts for the algorithms in the paper. Please contact Yishu Wang ([wangy@link.cuhk.edu.hk](wangy@link.cuhk.edu.hk)) if you have any questions about the code.

### Code structure

#### Data:
* `lianjia.RData`: Transaction-level dataset of housing prices in Beijing from Lianjia.

#### Master Files:
* `prediction.R`: Main file for performing spatial prediction in Section 3 where the best tuned models from `plm.knn.R`, `STKNNTune.R` and `GBMTune.R` are called.
* `pred_seq.R`: Main file for performing sequential forecast in Section 4 where the best tuned models from `plm.knn.R` and `GBMTune_seq.R` are called.

#### Machine Learning Methods:
* `plm.knn.R`: Tuning spatial k-Nearest Neighbor (KNN)
* `STKNNTune.R`: Tuning spatial-temporal KNN
* `GBMTune.R`: Tuning Gradient Boosting Machine (GBM, spatial version)
* `GBMTune_seq.R`: Tuning Gradient Boosting Machine (GBM, sequential version)

#### Visualization:
* `GBMplot.R`: Plotting Figure 3 "GBM prediction on coordinate raster"
