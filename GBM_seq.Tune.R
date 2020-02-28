library(caret)
library(doParallel)

load("lianjia.RData")

train_ind=which(lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=8))
lianjia.train=lianjia[train_ind,]
valid_ind=which(lianjia$tradeYear==2017 & lianjia$tradeMonth==9)

tune_ind=createDataPartition(1:nrow(lianjia.train),p=0.1)$Resample1
lianjia.tune=rbind(lianjia.train[tune_ind,],lianjia[valid_ind,])
valid_ind=which(lianjia.tune$tradeYear==2017 & lianjia.tune$tradeMonth==9)

gbmGrid=expand.grid(interaction.depth=(2:10)*5,
                    n.trees=(10:200)*100,
                    shrinkage=c(0.001,0.005,0.01,0.05,0.1),
                    n.minobsinnode=20)
gbmControl=trainControl(method="cv",number=1,savePredictions=T,
                        index=list((1:nrow(lianjia.tune))[-valid_ind]))

formula.GBM=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+Lng+Lat+tradeYQ+
  communityAverage

tc=Sys.time();cat(paste("Starting Tuning at:",tc),"\n")
registerDoParallel(24)
Tune.GBM=train(formula.GBM,data=lianjia.tune,
               method="gbm",distribution="gaussian",
               trControl=gbmControl,tuneGrid=gbmGrid,metric="Rsquared",
               verbose=F)
stopImplicitCluster()
tc=Sys.time()-tc;print(tc)

gbmTune=Tune.GBM$bestTune
print(gbmTune)
print(Tune.GBM$resample)


save(formula.GBM,gbmTune,Tune.GBM,file="GBMTune_seq.RData")



