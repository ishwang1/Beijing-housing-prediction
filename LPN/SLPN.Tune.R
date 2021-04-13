library(caret)
source("LPN/plm.localpoly.pred.R")

load("lianjia.RData")

p.grid=1
h.grid=data.frame(h1=seq(0.07,0.15,by=0.01),
                  h2=seq(0.07,0.15,by=0.01))
cl=24


lianjia.SLPN=list()
lianjia.SLPN$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.SLPN$X=cbind(lianjia.SLPN$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.SLPN$X=cbind(lianjia.SLPN$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.SLPN$X=cbind(lianjia.SLPN$X,
                     predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.SLPN$X=cbind(lianjia.SLPN$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.SLPN$X=cbind(lianjia.SLPN$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.SLPN$X=cbind(lianjia.SLPN$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-1])
lianjia.SLPN$X=cbind(lianjia.SLPN$X,communityAverage=lianjia$communityAverage)
lianjia.SLPN$X=as.matrix(lianjia.SLPN$X)
lianjia.SLPN$W=as.matrix(lianjia[,c("Lng","Lat")])


Tune.SLPN=plm.localpoly.tune.cv(y=lianjia$price,X=lianjia.SLPN$X,W=lianjia.SLPN$W,
                                h.grid=h.grid,p.grid=p.grid,fold=4,cl=cl)
print(Tune.SLPN)


save(Tune.SLPN,file="LPN/SLPNTune.RData")


