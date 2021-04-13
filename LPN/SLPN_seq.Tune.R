library(caret)
source("LPN/plm.localpoly.pred.R")

load("lianjia.RData")

p.grid=1
h.grid=data.frame(h1=seq(0.07,0.15,by=0.01),
                  h2=seq(0.07,0.15,by=0.01))
cl=24


lianjia=lianjia[lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=9),]
lianjia$tradeYQ=factor(lianjia$tradeYQ)
valid_ind=which(lianjia$tradeYear==2017 & lianjia$tradeMonth==9)

lianjia.SLPN.seq=list()
lianjia.SLPN.seq$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.SLPN.seq$X=cbind(lianjia.SLPN.seq$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.SLPN.seq$X=cbind(lianjia.SLPN.seq$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.SLPN.seq$X=cbind(lianjia.SLPN.seq$X,
                         predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.SLPN.seq$X=cbind(lianjia.SLPN.seq$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.SLPN.seq$X=cbind(lianjia.SLPN.seq$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.SLPN.seq$X=cbind(lianjia.SLPN.seq$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-1])
lianjia.SLPN.seq$X=cbind(lianjia.SLPN.seq$X,communityAverage=lianjia$communityAverage)
lianjia.SLPN.seq$X=as.matrix(lianjia.SLPN.seq$X)
lianjia.SLPN.seq$W=as.matrix(lianjia[,c("Lng","Lat")])


Tune.SLPN.seq=plm.localpoly.tune.oos(y=lianjia$price,X=lianjia.SLPN.seq$X,W=lianjia.SLPN.seq$W,
                                     h.grid=h.grid,p.grid=p.grid,valid.ind=valid_ind,cl=cl)
print(Tune.SLPN.seq)


save(Tune.SLPN.seq,file="LPN/SLPNTune_seq.RData")


