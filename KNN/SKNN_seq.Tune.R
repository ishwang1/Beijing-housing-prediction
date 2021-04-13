library(caret)
source("KNN/plm.knn.R")

load("lianjia.RData")
cl=24


lianjia=lianjia[lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=9),]
lianjia$tradeYQ=factor(lianjia$tradeYQ)
valid_ind=which(lianjia$tradeYear==2017 & lianjia$tradeMonth==9)

lianjia.SKNN.seq=list()
lianjia.SKNN.seq$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.SKNN.seq$X=cbind(lianjia.SKNN.seq$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.SKNN.seq$X=cbind(lianjia.SKNN.seq$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.SKNN.seq$X=cbind(lianjia.SKNN.seq$X,
                         predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.SKNN.seq$X=cbind(lianjia.SKNN.seq$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.SKNN.seq$X=cbind(lianjia.SKNN.seq$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.SKNN.seq$X=cbind(lianjia.SKNN.seq$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-1])
lianjia.SKNN.seq$X=cbind(lianjia.SKNN.seq$X,communityAverage=lianjia$communityAverage)
lianjia.SKNN.seq$X=as.matrix(lianjia.SKNN.seq$X)
lianjia.SKNN.seq$W=as.matrix(lianjia[,c("Lng","Lat")])


cat(paste("Starting tuning SKNN with Euclidean distance at",Sys.time()),"\n")
Tune.SKNN.L2.seq=plm.knn.tune.oos(y=lianjia$price,X=lianjia.SKNN.seq$X,W=lianjia.SKNN.seq$W,
                                  valid.ind=valid_ind,
                                  k.grid=seq(200,800,by=100),p=2,cl=cl)
print(Tune.SKNN.L2.seq)

cat(paste("Starting tuning SKNN with Manhattan distance at",Sys.time()),"\n")
Tune.SKNN.L1.seq=plm.knn.tune.oos(y=lianjia$price,X=lianjia.SKNN.seq$X,W=lianjia.SKNN.seq$W,
                                  valid.ind=valid_ind,
                                  k.grid=seq(500,800,by=100),p=1,cl=cl)
print(Tune.SKNN.L1.seq)


save(Tune.SKNN.L2.seq,Tune.SKNN.L1.seq,file="KNN/SKNNTune_seq.RData")


