library(caret)
source("NW/plm.NW.R")

load("lianjia.RData")

h.grid=data.frame(h1=seq(0.04,0.15,by=0.01),
                  h2=seq(0.04,0.15,by=0.01))
cl=24


lianjia=lianjia[lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=9),]
lianjia$tradeYQ=factor(lianjia$tradeYQ)
valid_ind=which(lianjia$tradeYear==2017 & lianjia$tradeMonth==9)

lianjia.SNW.seq=list()
lianjia.SNW.seq$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.SNW.seq$X=cbind(lianjia.SNW.seq$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNW.seq$X=cbind(lianjia.SNW.seq$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.SNW.seq$X=cbind(lianjia.SNW.seq$X,
                        predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.SNW.seq$X=cbind(lianjia.SNW.seq$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.SNW.seq$X=cbind(lianjia.SNW.seq$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNW.seq$X=cbind(lianjia.SNW.seq$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNW.seq$X=cbind(lianjia.SNW.seq$X,communityAverage=lianjia$communityAverage)
lianjia.SNW.seq$X=as.matrix(lianjia.SNW.seq$X)
lianjia.SNW.seq$W=as.matrix(lianjia[,c("Lng","Lat")])


Tune.SNW.seq=plm.NW.tune.oos(y=lianjia$price,X=lianjia.SNW.seq$X,W=lianjia.SNW.seq$W,
                             h.grid=h.grid,valid.ind=valid_ind,cl=cl)
print(Tune.SNW.seq)


save(Tune.SNW.seq,file="NW/SNWTune_seq.RData")


