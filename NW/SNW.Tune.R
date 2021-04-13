library(caret)
source("NW/plm.NW.R")

load("lianjia.RData")

h.grid=data.frame(h1=seq(0.04,0.15,by=0.01),
                  h2=seq(0.04,0.15,by=0.01))
cl=24


lianjia.SNW=list()
lianjia.SNW$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.SNW$X=cbind(lianjia.SNW$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNW$X=cbind(lianjia.SNW$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.SNW$X=cbind(lianjia.SNW$X,
                    predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.SNW$X=cbind(lianjia.SNW$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.SNW$X=cbind(lianjia.SNW$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNW$X=cbind(lianjia.SNW$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNW$X=cbind(lianjia.SNW$X,communityAverage=lianjia$communityAverage)
lianjia.SNW$X=as.matrix(lianjia.SNW$X)
lianjia.SNW$W=as.matrix(lianjia[,c("Lng","Lat")])


Tune.SNW=plm.NW.tune.cv(y=lianjia$price,X=lianjia.SNW$X,W=lianjia.SNW$W,
                        h.grid=h.grid,fold=4,cl=cl)
print(Tune.SNW)


save(Tune.SNW,file="NW/SNWTune.RData")


