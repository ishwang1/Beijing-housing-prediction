library(caret)
source("NW/plm.NW.R")

load("lianjia.RData")

h.grid=expand.grid(h1=seq(10,100,by=10),
                   h2=seq(0.05,0.4,by=0.05))
h.grid$h3=h.grid$h2

cl=24


lianjia.STNW=list()
lianjia.STNW$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.STNW$X=cbind(lianjia.STNW$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.STNW$X=cbind(lianjia.STNW$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.STNW$X=cbind(lianjia.STNW$X,
                     predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.STNW$X=cbind(lianjia.STNW$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.STNW$X=cbind(lianjia.STNW$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.STNW$X=cbind(lianjia.STNW$X,communityAverage=lianjia$communityAverage)
lianjia.STNW$X=as.matrix(lianjia.STNW$X)
lianjia.STNW$W=as.matrix(lianjia[,c("t_trade","Lng","Lat")])


Tune.STNW=plm.NW.tune.cv(y=lianjia$price,X=lianjia.STNW$X,W=lianjia.STNW$W,
                         h.grid=h.grid,fold=4,cl=cl)
print(Tune.STNW)


save(Tune.STNW,file="NW/STNWTune.RData")


