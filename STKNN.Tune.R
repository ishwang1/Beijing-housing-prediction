library(caret)


# Data

load("lianjia.RData")

lianjia.STKNN=list()
lianjia.STKNN$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.STKNN$X=cbind(lianjia.STKNN$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.STKNN$X=cbind(lianjia.STKNN$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.STKNN$X=cbind(lianjia.STKNN$X,
                      predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.STKNN$X=cbind(lianjia.STKNN$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.STKNN$X=cbind(lianjia.STKNN$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.STKNN$X=cbind(lianjia.STKNN$X,communityAverage=lianjia$communityAverage)
lianjia.STKNN$X=as.matrix(lianjia.STKNN$X)

lianjia.STKNN$W=as.matrix(lianjia[,c("t_trade","Lng","Lat")])


# KNN Tune

source("plm.knn.R")

Tune.STKNN=plm.knn.tune(k.grid=2:4*5,lambda.grid=cbind(c(0.0005,0.001,0.005,0.1,0.5,1,2,10),1,1),
                        y=lianjia$price,X=lianjia.STKNN$X,W=lianjia.STKNN$W,
                        fold=4,metric="R2",cl=24)

print(c(k=Tune.STKNN$k.best,lambda=Tune.STKNN$lambda.best))
print(Tune.STKNN$Performance)
save(lianjia.STKNN,Tune.STKNN,file="STKNNTune.RData")


