# Unfolding Beijing in a Hedonic Way

This is a public repository accompanying the paper
* [Wei Lin](http://www.weilinmetrics.com/), [Zhentao Shi](http://www.zhentaoshi.com/), Yishu Wang and Ting Hin Yan: [“Unfolding Beijing in a Hedonic Way”](https://www.researchgate.net/publication/339551353_Unfolding_Beijing_in_a_Hedonic_Way).

for access of the data and R scripts. Please contact Yishu Wang ([wangy@link.cuhk.edu.hk](wangy@link.cuhk.edu.hk)) if you have any questions about the code.

### Code structure

#### Data:
* `lianjia.RData`: Transaction-level dataset of housing prices in Beijing from Lianjia.

#### Master Files:
* `prediction.R`: Main file for performing spatial prediction in Section 3.
* `pred_seq.R`: Main file for performing sequential forecast in Section 4.

#### KNN Tuning: 

* `KNN/plm.knn.R`: Functions of partial linear k-Nearest Neighbor (KNN)
* `KNN/SKNN.Tune.R`: Tuning spatial KNN
* `KNN/STKNN.Tune.R`: Tuning spatial-temporal KNN
* `KNN/SKNN_seq.Tune.R`: Tuning sequential spatial KNN

#### NW Tuning: 

* `NW/plm.NW.R`: Functions of partial linear Nadaraya-Watson (NW)
* `NW/SNW.Tune.R`: Tuning spatial NW
* `NW/STNW.Tune.R`: Tuning spatial-temporal NW
* `NW/SNW_seq.Tune.R`: Tuning sequential spatial NW

#### LPN Tuning: 

* `LPN/plm.localpoly.pred`: Functions of partial linear Local Polynomial (LPN)
* `LPN/SLPN.Tune.R`: Tuning spatial LPN
* `LPN/STLPN.Tune.R`: Tuning spatial-temporal LPN
* `LPN/SLPN_seq.Tune.R`: Tuning sequential spatial LPN

#### RF Tuning:

* `RF/RF.Tune.R`: Tuning Random Forests (RF, spatial version)
* `RF/RF_seq.Tune.R`: Tuning Random Forests (RF, sequential version)

#### GBM Tuning:

* `GBM/GBM.Tune.R`: Tuning Gradient Boosting Machine (GBM, spatial version)
* `GBM/GBM_seq.Tune.R`: Tuning Gradient Boosting Machine (GBM, sequential version)

#### Visualization:
* `GBMplot.R`: Plotting Figure 3 "GBM prediction on coordinate raster"

Remark: If you want to save time from tuning parameters and trust our tuning results, you can directly run `prediction.R` and `pred_seq.R` to replicate our main results. 

