library(caret)

source("KNN/plm.knn.R")
source("NW/plm.NW.R")
source("LPN/plm.localpoly.pred.R")

load("lianjia.RData")

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

# STNP formula
lianjia.STNP=list()
lianjia.STNP$X=lianjia[,c("square","livingRoom","drawingRoom","kitchen","bathRoom")]
lianjia.STNP$X=cbind(lianjia.STNP$X,predict(dummyVars(~floor_type,data=lianjia),newdata=lianjia)[,-1])
lianjia.STNP$X=cbind(lianjia.STNP$X,lianjia[,c("floor_total","elevator","ladderRatio")])
lianjia.STNP$X=cbind(lianjia.STNP$X,
                     predict(dummyVars(~renovationCondition+buildingType+buildingStructure,data=lianjia),newdata=lianjia)[,-c(1,4,8)])
lianjia.STNP$X=cbind(lianjia.STNP$X,lianjia[,c("age","DOM","followers","fiveYearsProperty","subway")])
lianjia.STNP$X=cbind(lianjia.STNP$X,predict(dummyVars(~district,data=lianjia),newdata=lianjia)[,-1])
lianjia.STNP$X=cbind(lianjia.STNP$X,communityAverage=lianjia$communityAverage)
lianjia.STNP$X=as.matrix(lianjia.STNP$X)
lianjia.STNP$W=as.matrix(lianjia[,c("t_trade","Lng","Lat")])

# Tree formula without coordinates
formula.tree.noc=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+t_trade+
  communityAverage

# Tree formula with coordinates
formula.tree=price~
  square+livingRoom+drawingRoom+kitchen+bathRoom+
  floor_type+floor_total+elevator+ladderRatio+
  renovationCondition+buildingType+buildingStructure+
  age+DOM+followers+fiveYearsProperty+
  subway+district+Lng+Lat+t_trade+
  communityAverage


# Training sample
set.seed(2021)
train_ind=createDataPartition(1:nrow(lianjia),p=0.75)$Resample1


# OLS
cat(paste("Starting OLS at",Sys.time()),"\n")
model.OLS=lm(formula.OLS,data=lianjia[train_ind,])
pred.OLS=predict(model.OLS,newdata=lianjia[-train_ind,])

# SKNN with L1 distance
cat(paste("Starting SKNN.L1 at",Sys.time()),"\n")
model.SKNN.L1=plm.knn(k=400,
                      y=lianjia$price[train_ind],X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                      X.new=lianjia.SNP$X[-train_ind,],W.new=lianjia.SNP$W[-train_ind,],
                      p=1,cl=cl)
pred.SKNN.L1=model.SKNN.L1$pred.out

# STKNN with L1 distance
cat(paste("Starting STKNN.L1 at",Sys.time()),"\n")
model.STKNN.L1=plm.knn(k=15,
                       y=lianjia$price[train_ind],X=lianjia.STNP$X[train_ind,],W=lianjia.STNP$W[train_ind,],
                       X.new=lianjia.STNP$X[-train_ind,],W.new=lianjia.STNP$W[-train_ind,],
                       lambda=c(0.0005,1,1),p=1,cl=cl)
pred.STKNN.L1=model.STKNN.L1$pred.out

# SKNN with L2 distance
cat(paste("Starting SKNN.L2 at",Sys.time()),"\n")
model.SKNN.L2=plm.knn(k=200,
                      y=lianjia$price[train_ind],X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                      X.new=lianjia.SNP$X[-train_ind,],W.new=lianjia.SNP$W[-train_ind,],
                      p=2,cl=cl)
pred.SKNN.L2=model.SKNN.L2$pred.out

# STKNN with L2 distance
cat(paste("Starting STKNN.L2 at",Sys.time()),"\n")
model.STKNN.L2=plm.knn(k=15,
                       y=lianjia$price[train_ind],X=lianjia.STNP$X[train_ind,],W=lianjia.STNP$W[train_ind,],
                       X.new=lianjia.STNP$X[-train_ind,],W.new=lianjia.STNP$W[-train_ind,],
                       lambda=c(0.0005,1,1),p=2,cl=cl)
pred.STKNN.L2=model.STKNN.L2$pred.out

# SNW
cat(paste("Starting SNW at",Sys.time()),"\n")
model.SNW=plm.NW(h=c(0.04,0.04),
                 y=lianjia$price[train_ind],X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                 X.new=lianjia.SNP$X[-train_ind,],W.new=lianjia.SNP$W[-train_ind,],
                 cl=cl)
pred.SNW=model.SNW$pred.out

# STNW
cat(paste("Starting STNW at",Sys.time()),"\n")
model.STNW=plm.NW(h=c(30,0.05,0.05),
                  y=lianjia$price[train_ind],X=lianjia.STNP$X[train_ind,],W=lianjia.STNP$W[train_ind,],
                  X.new=lianjia.STNP$X[-train_ind,],W.new=lianjia.STNP$W[-train_ind,],
                  cl=cl)
pred.STNW=model.STNW$pred.out

# SLPN.1
cat(paste("Starting SLPN.1 at",Sys.time()),"\n")
pred.SLPN.1=plm.localpoly.pred(y=lianjia$price[train_ind],
                               X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                               X.new=lianjia.SNP$X[-train_ind,],W.new=lianjia.SNP$W[-train_ind,],
                               h=c(0.07,0.07),p=1,cl=cl)

# STLPN.1
cat(paste("Starting STLPN.1 at",Sys.time()),"\n")
pred.STLPN.1=plm.localpoly.pred(y=lianjia$price[train_ind],
                                X=lianjia.STNP$X[train_ind,],W=lianjia.STNP$W[train_ind,],
                                X.new=lianjia.STNP$X[-train_ind,],W.new=lianjia.STNP$W[-train_ind,],
                                h=c(50,0.3,0.3),p=1,cl=cl)

# SLPN.2
cat(paste("Starting SLPN.2 at",Sys.time()),"\n")
pred.SLPN.2=plm.localpoly.pred(y=lianjia$price[train_ind],
                               X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                               X.new=lianjia.SNP$X[-train_ind,],W.new=lianjia.SNP$W[-train_ind,],
                               h=c(0.09,0.09),p=2,cl=cl)

# SLPN.3
cat(paste("Starting SLPN.3 at",Sys.time()),"\n")
pred.SLPN.3=plm.localpoly.pred(y=lianjia$price[train_ind],
                               X=lianjia.SNP$X[train_ind,],W=lianjia.SNP$W[train_ind,],
                               X.new=lianjia.SNP$X[-train_ind,],W.new=lianjia.SNP$W[-train_ind,],
                               h=c(0.11,0.11),p=3,cl=cl)

# RF without Coordinates
cat(paste("Starting RF without coordinates at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
model.RF.noc=train(formula.tree.noc,data=lianjia[train_ind,],
                   method="rf",distribution="gaussian",
                   tuneGrid=data.frame(mtry=29),nodesize=20,maxnodes=10500,
                   verbose=F)
pred.RF.noc=predict(model.RF.noc,newdata=lianjia[-train_ind,])
doParallel::stopImplicitCluster()

# RF with Coordinates
cat(paste("Starting RF with coordinates at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
model.RF=train(formula.tree,data=lianjia[train_ind,],
               method="rf",distribution="gaussian",
               tuneGrid=data.frame(mtry=30),nodesize=20,maxnodes=9800,
               verbose=F)
pred.RF=predict(model.RF,newdata=lianjia[-train_ind,])
doParallel::stopImplicitCluster()

# GBM without Coordinates
cat(paste("Starting GBM without coordinates at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
model.GBM.noc=train(formula.tree.noc,data=lianjia[train_ind,],
                    method="gbm",distribution="gaussian",
                    tuneGrid=data.frame(n.trees=12000,
                                        interaction.depth=25,
                                        shrinkage=0.005,
                                        n.minobsinnode=20),
                    verbose=F)
pred.GBM.noc=predict(model.GBM.noc,newdata=lianjia[-train_ind,])
doParallel::stopImplicitCluster()

# GBM with Coordinates
cat(paste("Starting GBM with coordinates at",Sys.time()),"\n")
doParallel::registerDoParallel(cl)
model.GBM=train(formula.tree,data=lianjia[train_ind,],
                method="gbm",distribution="gaussian",
                tuneGrid=data.frame(n.trees=19400,
                                    interaction.depth=25,
                                    shrinkage=0.005,
                                    n.minobsinnode=20),
                verbose=F)
pred.GBM=predict(model.GBM,newdata=lianjia[-train_ind,])
doParallel::stopImplicitCluster()


# Result
result=data.frame(target=lianjia$price[-train_ind],
                  OLS=pred.OLS,
                  SKNN.L1=pred.SKNN.L1,STKNN.L1=pred.STKNN.L1,
                  SKNN.L2=pred.SKNN.L2,STKNN.L2=pred.STKNN.L2,
                  SNW=pred.SNW,STNW=pred.STNW,
                  SLPN.1=pred.SLPN.1,STLPN.1=pred.STLPN.1,
                  SLPN.2=pred.SLPN.2,SLPN.3=pred.SLPN.3,
                  RF.noc=pred.RF.noc,RF=pred.RF,
                  GBM.noc=pred.GBM.noc,GBM=pred.GBM)


save(result,file="result.RData")
save(model.RF,model.GBM,file="model.tree.RData")
cat("Spatial prediction finished!")


