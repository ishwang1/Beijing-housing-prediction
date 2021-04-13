library(caret)
source("LPN/plm.localpoly.pred.R")

load("lianjia.RData")

p.grid=1
h.grid=expand.grid(h1=seq(50,100,by=10),
                   h2=seq(0.3,0.4,by=0.05))
h.grid$h3=h.grid$h2

cl=24


lianjia.STLPN=list()
lianjia.STLPN$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.STLPN$X=cbind(lianjia.STLPN$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.STLPN$X=cbind(lianjia.STLPN$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.STLPN$X=cbind(lianjia.STLPN$X,
                     predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.STLPN$X=cbind(lianjia.STLPN$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.STLPN$X=cbind(lianjia.STLPN$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.STLPN$X=cbind(lianjia.STLPN$X,communityAverage=lianjia$communityAverage)
lianjia.STLPN$X=as.matrix(lianjia.STLPN$X)
lianjia.STLPN$W=as.matrix(lianjia[,c("t_trade","Lng","Lat")])


Tune.STLPN=plm.localpoly.tune.cv(y=lianjia$price,X=lianjia.STLPN$X,W=lianjia.STLPN$W,
                                 h.grid=h.grid,p.grid=p.grid,fold=4,cl=cl)
print(Tune.STLPN)


save(Tune.STLPN,file="LPN/STLPNTune.RData")


