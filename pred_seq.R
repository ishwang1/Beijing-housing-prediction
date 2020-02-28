library(caret)
library(doParallel)

source("plm.knn.R")
load("lianjia.RData")

YQM=aggregate(lianjia$id,by=list(lianjia$tradeYear,lianjia$tradeQuarter,lianjia$tradeMonth),FUN=length)
colnames(YQM)=c("Year","Quarter","Month","n")
YQM=YQM[order(YQM$Year,YQM$Month),]

train_ind=which(lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=11))
test_ind=which(lianjia$tradeYear==2017 & lianjia$tradeMonth==12)


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
lianjia.SKNN$X=cbind(lianjia.SKNN$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-c(1,33)])
lianjia.SKNN$X=cbind(lianjia.SKNN$X,communityAverage=lianjia$communityAverage)
lianjia.SKNN$X=as.matrix(lianjia.SKNN$X)
lianjia.SKNN$W=as.matrix(lianjia[,c("Lng","Lat")])

# GBM
load("GBMTune_seq.RData")


pred_seq.lianjia=function(cl=24) {
  
  # OLS
  cat(paste("Starting OLS at:",Sys.time()),"\n")
  model.OLS=lm(formula.OLS,data=lianjia[train_ind,])
  pred.OLS=predict(model.OLS,newdata=lianjia[test_ind,])
  
  # log OLS
  cat(paste("Starting log OLS at:",Sys.time()),"\n")
  model.log=lm(formula.log,data=lianjia[train_ind,])
  pred.log=exp(predict(model.log,newdata=lianjia[test_ind,]))
  
  # SKNN
  cat(paste("Starting SKNN at:",Sys.time()),"\n")
  model.SKNN=plm.knn(k=200,cl=cl,
                     y=lianjia$price[train_ind],X=lianjia.SKNN$X[train_ind,],W=lianjia.SKNN$W[train_ind,],
                     X.new=lianjia.SKNN$X[test_ind,],W.new=lianjia.SKNN$W[test_ind,])
  pred.SKNN=model.SKNN$pred.plm
  
  # GBM
  cat(paste("Starting GBM at:",Sys.time()),"\n")
  registerDoParallel(cl)
  model.GBM=train(formula.GBM,data=lianjia[train_ind,],
                  method="gbm",distribution="gaussian",tuneGrid=gbmTune,
                  verbose=F)
  pred.GBM=predict(model.GBM,newdata=lianjia[test_ind,])
  stopImplicitCluster()
  
  return(data.frame(target=lianjia$price[test_ind],
                    OLS=pred.OLS,
                    OLS.log=pred.log,
                    SKNN=pred.SKNN,
                    GBM=pred.GBM))
}

tc=Sys.time();cat(paste("Prediction starting at:",tc),"\n")
result_seq=pred_seq.lianjia()
tc=Sys.time()-tc;print(tc)
save(result_seq,YQM,file="result_seq.RData")


