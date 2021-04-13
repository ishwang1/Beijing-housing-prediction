library(caret)
source("KNN/plm.knn.R")

load("lianjia.RData")
cl=24

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


cat(paste("Starting tuning STKNN with Euclidean distance at",Sys.time()),"\n")
Tune.STKNN.L2=plm.knn.tune.cv(y=lianjia$price,X=lianjia.STKNN$X,W=lianjia.STKNN$W,
                              k.grid=seq(10,30,by=5),lambda.grid=cbind(c(0.0005,0.001,0.005,0.1,0.5,1,2),1,1),
                              p=2,fold=4,cl=cl)
print(Tune.STKNN.L2)

cat(paste("Starting tuning STKNN with Manhattan distance at",Sys.time()),"\n")
Tune.STKNN.L1=plm.knn.tune.cv(y=lianjia$price,X=lianjia.STKNN$X,W=lianjia.STKNN$W,
                              k.grid=seq(15,30,by=5),lambda.grid=cbind(c(0.0005,0.001,0.005,0.1,0.5,1,2),1,1),
                              p=1,fold=4,cl=cl)
print(Tune.STKNN.L1)


save(Tune.STKNN.L2,Tune.STKNN.L1,file="KNN/STKNNTune.RData")


