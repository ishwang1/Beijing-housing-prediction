library(caret)
library(doParallel)

source("plm.knn.R")
load("lianjia.RData")


# OLS
formula.OLS=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+dist.center+tradeYQ+
  communityAverage

# log OLS
lianjia$lprice=log(lianjia$price)
formula.log=lprice~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+dist.center+tradeYQ+
  communityAverage

# SKNN
lianjia.SKNN=list()
lianjia.SKNN$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.SKNN$X=cbind(lianjia.SKNN$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.SKNN$X=cbind(lianjia.SKNN$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.SKNN$X=cbind(lianjia.SKNN$X,
                     predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.SKNN$X=cbind(lianjia.SKNN$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.SKNN$X=cbind(lianjia.SKNN$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.SKNN$X=cbind(lianjia.SKNN$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-1])
lianjia.SKNN$X=cbind(lianjia.SKNN$X,communityAverage=lianjia$communityAverage)
lianjia.SKNN$X=as.matrix(lianjia.SKNN$X)
lianjia.SKNN$W=as.matrix(lianjia[,c("Lng","Lat")])

# STKNN
load("STKNNTune.RData")

# GBM
load("GBMTune.RData")


pred.lianjia=function(seed=NULL,GBM.model.save=F,cl=24) {
  
  set.seed(seed)
  train_ind=createDataPartition(1:nrow(lianjia),p=0.75)$Resample1
  
  # OLS
  cat(paste("Starting OLS at:",Sys.time()),"\n")
  model.OLS=lm(formula.OLS,data=lianjia[train_ind,])
  pred.OLS=predict(model.OLS,newdata=lianjia[-train_ind,])
  
  # log OLS
  cat(paste("Starting log OLS at:",Sys.time()),"\n")
  model.log=lm(formula.log,data=lianjia[train_ind,])
  pred.log=exp(predict(model.log,newdata=lianjia[-train_ind,]))
  
  # SKNN
  cat(paste("Starting SKNN at:",Sys.time()),"\n")
  model.SKNN=plm.knn(k=200,cl=cl,
                     y=lianjia$price[train_ind],X=lianjia.SKNN$X[train_ind,],W=lianjia.SKNN$W[train_ind,],
                     X.new=lianjia.SKNN$X[-train_ind,],W.new=lianjia.SKNN$W[-train_ind,])
  pred.SKNN=model.SKNN$pred.plm
  
  # STKNN
  cat(paste("Starting STKNN at:",Sys.time()),"\n")
  model.STKNN=plm.knn(k=Tune.STKNN$k.best,lambda=Tune.STKNN$lambda.best,cl=cl,
                      y=lianjia$price[train_ind],X=lianjia.STKNN$X[train_ind,],W=lianjia.STKNN$W[train_ind,],
                      X.new=lianjia.STKNN$X[-train_ind,],W.new=lianjia.STKNN$W[-train_ind,])
  pred.STKNN=model.STKNN$pred.plm
  
  # GBM without Coorniates
  cat(paste("Starting GBM without Coordinates at:",Sys.time()),"\n")
  registerDoParallel(cl)
  model.GBM.noc=train(formula.GBM.noc,data=lianjia[train_ind,],
                      method="gbm",distribution="gaussian",tuneGrid=gbmTune.noc,
                      verbose=F)
  pred.GBM.noc=predict(model.GBM.noc,newdata=lianjia[-train_ind,])
  stopImplicitCluster()
  
  # GBM with Coorniates
  cat(paste("Starting GBM with Coordinates at:",Sys.time()),"\n")
  registerDoParallel(cl)
  model.GBM=train(formula.GBM,data=lianjia[train_ind,],
                  method="gbm",distribution="gaussian",tuneGrid=gbmTune,
                  verbose=F)
  pred.GBM=predict(model.GBM,newdata=lianjia[-train_ind,])
  stopImplicitCluster()
  
  if(GBM.model.save==T) {
    return(list(prediction=data.frame(target=lianjia$price[-train_ind],
                                      OLS=pred.OLS,
                                      OLS.log=pred.log,
                                      SKNN=pred.SKNN,
                                      STKNN=pred.STKNN,
                                      GBM.noc=pred.GBM.noc,
                                      GBM=pred.GBM),
                GBM.model=model.GBM))
  } else {
    return(data.frame(target=lianjia$price[-train_ind],
                      OLS=pred.OLS,
                      OLS.log=pred.log,
                      SKNN=pred.SKNN,
                      STKNN=pred.STKNN,
                      GBM.noc=pred.GBM.noc,
                      GBM=pred.GBM))
  }
}

tc=Sys.time();cat(paste("Prediction starting at:",tc),"\n")
result=pred.lianjia(GBM.model.save=T)
tc=Sys.time()-tc;print(tc)
save(result,file="result.RData")


