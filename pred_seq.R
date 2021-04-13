library(caret)

source("KNN/plm.knn.R")
source("NW/plm.NW.R")
source("LPN/plm.localpoly.pred.R")

load("lianjia.RData")


# Training and testing sample
lianjia=lianjia[lianjia$tradeYear<=2017,]
lianjia$tradeYQ=factor(lianjia$tradeYQ)
train_ind=which(lianjia$tradeYear<=2016 | (lianjia$tradeYear==2017 & lianjia$tradeMonth<=11))
test_ind=which(lianjia$tradeYear==2017 & lianjia$tradeMonth==12)

cl=24


# OLS formula
formula.OLS=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+dist.center+tradeYQ+
  communityAverage

# SNP formula
lianjia.SNP=list()
lianjia.SNP$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.SNP$X=cbind(lianjia.SNP$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNP$X=cbind(lianjia.SNP$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.SNP$X=cbind(lianjia.SNP$X,
                    predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.SNP$X=cbind(lianjia.SNP$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.SNP$X=cbind(lianjia.SNP$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNP$X=cbind(lianjia.SNP$X,predict(dummyVars(~tradeYQ,data=lianjia),newdata=lianjia)[,-1])
lianjia.SNP$X=cbind(lianjia.SNP$X,communityAverage=lianjia$communityAverage)
lianjia.SNP$X=as.matrix(lianjia.SNP$X)
lianjia.SNP$W=as.matrix(lianjia[,c("Lng","Lat")])

# Tree formula
formula.tree=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+Lng+Lat+tradeYQ+
  communityAverage


# OLS
cat(paste("Starting OLS at",Sys.time()),"\n")
model.OLS=lm(formula.OLS,data=lianjia[train_ind,])
pred.OLS=predict(model.OLS,newdata=lianjia[test_ind,])

# SKNN with L1 distance
cat(paste("Starting SKNN.L1 at",Sys.time()),"\n")
model.SKNN.L1=plm.knn(k=500,
                      y=lianjia$price[train_ind],X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                      X.new=lianjia.SNP$X[test_ind,],W.new=lianjia.SNP$W[test_ind,],
                      p=1,cl=cl)
pred.SKNN.L1=model.SKNN.L1$pred.out

# SKNN with L2 distance
cat(paste("Starting SKNN.L2 at",Sys.time()),"\n")
model.SKNN.L2=plm.knn(k=200,
                      y=lianjia$price[train_ind],X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                      X.new=lianjia.SNP$X[test_ind,],W.new=lianjia.SNP$W[test_ind,],
                      p=2,cl=cl)
pred.SKNN.L2=model.SKNN.L2$pred.out

# SNW
cat(paste("Starting SNW at",Sys.time()),"\n")
model.SNW=plm.NW(h=c(0.04,0.04),
                 y=lianjia$price[train_ind],X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                 X.new=lianjia.SNP$X[test_ind,],W.new=lianjia.SNP$W[test_ind,],
                 cl=cl)
pred.SNW=model.SNW$pred.out

# SLPN.1
cat(paste("Starting SLPN.1 at",Sys.time()),"\n")
pred.SLPN.1=plm.localpoly.pred(y=lianjia$price[train_ind],
                               X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                               X.new=lianjia.SNP$X[test_ind,],W.new=lianjia.SNP$W[test_ind,],
                               h=c(0.07,0.07),p=1,cl=cl)

# SLPN.2
cat(paste("Starting SLPN.2 at",Sys.time()),"\n")
pred.SLPN.2=plm.localpoly.pred(y=lianjia$price[train_ind],
                               X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                               X.new=lianjia.SNP$X[test_ind,],W.new=lianjia.SNP$W[test_ind,],
                               h=c(0.09,0.09),p=2,cl=cl)

# SLPN.3
cat(paste("Starting SLPN.3 at",Sys.time()),"\n")
pred.SLPN.3=plm.localpoly.pred(y=lianjia$price[train_ind],
                               X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                               X.new=lianjia.SNP$X[test_ind,],W.new=lianjia.SNP$W[test_ind,],
                               h=c(0.11,0.11),p=3,cl=cl)

# RF
cat(paste("Starting RF at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
model.RF=train(formula.tree,data=lianjia[train_ind,],
               method="rf",distribution="gaussian",
               tuneGrid=data.frame(mtry=38),nodesize=20,maxnodes=10100,
               verbose=F)
pred.RF=predict(model.RF,newdata=lianjia[test_ind,])
doParallel::stopImplicitCluster()

# GBM
cat(paste("Starting GBM at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
model.GBM=train(formula.tree,data=lianjia[train_ind,],
                method="gbm",distribution="gaussian",
                tuneGrid=data.frame(n.trees=16900,
                                    interaction.depth=15,
                                    shrinkage=0.005,
                                    n.minobsinnode=20),
                verbose=F)
pred.GBM=predict(model.GBM,newdata=lianjia[test_ind,])
doParallel::stopImplicitCluster()


result_seq=data.frame(target=lianjia$price[test_ind],
                      OLS=pred.OLS,
                      SKNN.L1=pred.SKNN.L1,SKNN.L2=pred.SKNN.L2,
                      SNW=pred.SNW,
                      SLPN.1=pred.SLPN.1,SLPN.2=pred.SLPN.2,SLPN.3=pred.SLPN.3,
                      RF=pred.RF,GBM=pred.GBM)


save(result_seq,file="result_seq.RData")
cat("Sequential prediction finished!")


