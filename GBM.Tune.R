library(caret)
library(doParallel)

load("lianjia.RData")

tune_ind=createDataPartition(1:nrow(lianjia),p=0.1)$Resample1
gbmGrid=expand.grid(interaction.depth=(2:10)*5,
                    n.trees=(10:200)*100,
                    shrinkage=c(0.001,0.005,0.01,0.05,0.1),
                    n.minobsinnode=20)
gbmControl=trainControl(method="cv",number=5)


# Without Coordinates

formula.GBM.noc=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+t_trade+
  communityAverage

tc=Sys.time();cat(paste("Starting Tuning without Coordinates at:",tc),"\n")
registerDoParallel(24)
Tune.GBM.noc=train(formula.GBM.noc,data=lianjia[tune_ind,],
                   method="gbm",distribution="gaussian",
                   trControl=gbmControl,tuneGrid=gbmGrid,metric="Rsquared",
                   verbose=F)
stopImplicitCluster()
tc=Sys.time()-tc;print(tc)

gbmTune.noc=Tune.GBM.noc$bestTune
print(gbmTune.noc)
print(Tune.GBM.noc$resample)


# With Coordinates

formula.GBM=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+Lng+Lat+t_trade+
  communityAverage

tc=Sys.time();cat(paste("Starting Tuning with Coordinates at:",tc),"\n")
registerDoParallel(24)
Tune.GBM=train(formula.GBM,data=lianjia[tune_ind,],
               method="gbm",distribution="gaussian",
               trControl=gbmControl,tuneGrid=gbmGrid,metric="Rsquared",
               verbose=F)
stopImplicitCluster()
tc=Sys.time()-tc;print(tc)

gbmTune=Tune.GBM$bestTune
print(gbmTune)
print(Tune.GBM$resample)


save(formula.GBM.noc,gbmTune.noc,formula.GBM,gbmTune,file="GBMTune.RData")


