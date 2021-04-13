library(caret)

load("lianjia.RData")
tune_ind=createDataPartition(1:nrow(lianjia),p=0.1)$Resample1

ntnode.grid=seq(8000,12000,by=100)
rfGrid=data.frame(mtry=10:50)
rfControl=trainControl(method="cv",number=5)
cl=24


# Without Coordinates

formula.RF.noc=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+t_trade+
  communityAverage

cat(paste("Starting tuning without coordinates at",Sys.time()),"\n")
rfTune.noc=data.frame()
for (i in 1:length(ntnode.grid)) {
  cat(paste("Starting tuning number of terminal nodes",i,"at",Sys.time()),"\n")
  doParallel::registerDoParallel(cl)
  Tune.RF.noc=train(formula.RF.noc,data=lianjia[tune_ind,],
                    method="rf",distribution="gaussian",
                    trControl=rfControl,tuneGrid=rfGrid,
                    nodesize=20,maxnodes=ntnode.grid[i],
                    metric="Rsquared",verbose=F)
  doParallel::stopImplicitCluster()
  rfTune.noc=rbind(rfTune.noc,
                   cbind(ntnode=ntnode.grid[i],Tune.RF.noc$results[,1:3]))
}
rfbestTune.noc=rfTune.noc[which.max(rfTune.noc$Rsquared),c("ntnode","mtry")]

print(rfbestTune.noc)
print(rfTune.noc)


# With Coordinates

formula.RF=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+Lng+Lat+t_trade+
  communityAverage

cat(paste("Starting tuning with coordinates at",Sys.time()),"\n")
rfTune=data.frame()
for (i in 1:length(ntnode.grid)) {
  cat(paste("Starting tuning number of terminal nodes",i,"at",Sys.time()),"\n")
  doParallel::registerDoParallel(cl)
  Tune.RF=train(formula.RF,data=lianjia[tune_ind,],
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


save(rfbestTune.noc,rfbestTune,file="RF/RFTune.RData")


