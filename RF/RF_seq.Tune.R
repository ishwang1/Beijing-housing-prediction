library(caret)


load("lianjia.RData")

lianjia=lianjia[lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=9),]
lianjia$tradeYQ=factor(lianjia$tradeYQ)

train_ind=which(lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=8))
lianjia.train=lianjia[train_ind,]
valid_ind=which(lianjia$tradeYear==2017 & lianjia$tradeMonth==9)

tune_ind=createDataPartition(1:nrow(lianjia.train),p=0.1)$Resample1
lianjia.tune=rbind(lianjia.train[tune_ind,],lianjia[valid_ind,])
valid_ind=which(lianjia.tune$tradeYear==2017 & lianjia.tune$tradeMonth==9)


ntnode.grid=seq(8000,12000,by=100)
rfGrid=data.frame(mtry=10:50)
rfControl=trainControl(method="cv",number=1,savePredictions=T,
                       index=list((1:nrow(lianjia.tune))[-valid_ind]))
cl=24


formula.RF=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+Lng+Lat+tradeYQ+
  communityAverage

rfTune=data.frame()
for (i in 1:length(ntnode.grid)) {
  cat(paste("Starting tuning number of terminal nodes",i,"at",Sys.time()),"\n")
  doParallel::registerDoParallel(cl)
  Tune.RF=train(formula.RF,data=lianjia.tune,
                method="rf",distribution="gaussian",
                trControl=rfControl,tuneGrid=rfGrid,
                nodesize=20,maxnodes=ntnode.grid[i],
                metric="Rsquared",verbose=F)
  doParallel::stopImplicitCluster()
  rfTune=rbind(rfTune,
               cbind(ntnode=ntnode.grid[i],Tune.RF$results[,1:3]))
}
rfbestTune=rfTune[which.max(rfTune$Rsquared),c("ntnode","mtry")]

print(rfbestTune)
print(rfTune)


save(rfbestTune,file="RF/RFTune_seq.RData")


