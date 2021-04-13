library(caret)

load("lianjia.RData")
tune_ind=createDataPartition(1:nrow(lianjia),p=0.1)$Resample1

gbmGrid=expand.grid(interaction.depth=seq(10:50,by=5),
                    n.trees=seq(10000,20000,by=100),
                    shrinkage=c(0.001,0.005,0.01,0.05,0.1),
                    n.minobsinnode=20)
gbmControl=trainControl(method="cv",number=5)
cl=24


# Without Coordinates

formula.GBM.noc=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+t_trade+
  communityAverage

cat(paste("Starting tuning without coordinates at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
Tune.GBM.noc=train(formula.GBM.noc,data=lianjia[tune_ind,],
                   method="gbm",distribution="gaussian",
                   trControl=gbmControl,tuneGrid=gbmGrid,metric="Rsquared",
                   verbose=F)
doParallel::stopImplicitCluster()

gbmTune.noc=Tune.GBM.noc$bestTune
print(gbmTune.noc)
print(Tune.GBM.noc$results)


# With Coordinates

formula.GBM=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+Lng+Lat+t_trade+
  communityAverage

cat(paste("Starting tuning with coordinates at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
Tune.GBM=train(formula.GBM,data=lianjia[tune_ind,],
               method="gbm",distribution="gaussian",
               trControl=gbmControl,tuneGrid=gbmGrid,metric="Rsquared",
               verbose=F)
doParallel::stopImplicitCluster()

gbmTune=Tune.GBM$bestTune
print(gbmTune)
print(Tune.GBM$resample)


save(gbmTune.noc,gbmTune,file="GBM/GBMTune.RData")


