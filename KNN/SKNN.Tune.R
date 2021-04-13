library(caret)
source("KNN/plm.knn.R")

load("lianjia.RData")
cl=24


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


cat(paste("Starting tuning SKNN with Euclidean distance at",Sys.time()),"\n")
Tune.SKNN.L2=plm.knn.tune.cv(y=lianjia$price,X=lianjia.SKNN$X,W=lianjia.SKNN$W,
                             k.grid=seq(200,800,by=100),p=2,fold=4,cl=cl)
print(Tune.SKNN.L2)

cat(paste("Starting tuning SKNN with Manhattan distance at",Sys.time()),"\n")
Tune.SKNN.L1=plm.knn.tune.cv(y=lianjia$price,X=lianjia.SKNN$X,W=lianjia.SKNN$W,
                             k.grid=seq(400,800,by=100),p=1,fold=4,cl=cl)
print(Tune.SKNN.L1)


save(Tune.SKNN.L2,Tune.SKNN.L1,file="KNN/SKNNTune.RData")


