plm.knn=function(k,y,X,W,lambda=1,X.new=NULL,W.new=NULL,cl=1) {
  
  library(foreach)
  library(doParallel)
  
  n=length(y)
  if(length(lambda)==1) {lambda=rep(lambda,ncol(W))}
  Lambda=matrix(rep(lambda,n),nrow=n,byrow=T)
  
  nb=function(w) {
    d=sqrt(rowSums(((W-matrix(rep(w,n),nrow=n,byrow=T))*Lambda)^2))
    d_k=sort(d)[k]
    nb_ind=which(d<=d_k)
    return(nb_ind)
  }
  
  knn.mean=function(i,W.nb) {
    nb_ind=nb(W.nb[i,])
    y_bar=mean(y[nb_ind])
    x_bar=colMeans(X[nb_ind,])
    return(c(y_bar,x_bar))
  }
  
  if(cl>1) {
    registerDoParallel(cl)
    M=foreach(i=1:n,.combine=rbind) %dopar% {knn.mean(i,W)}
    stopImplicitCluster()
  } else {
    M=foreach(i=1:n,.combine=rbind) %do% {knn.mean(i,W)}
  }
  rownames(M)=NULL
  
  y_demean=y-M[,1];X_demean=X-M[,-1]
  beta=as.vector(solve(t(X_demean)%*%X_demean)%*%t(X_demean)%*%y_demean)
  
  if(is.null(X.new)|is.null(W.new)) {
    
    y_hat=as.vector(X_demean%*%beta)+M[,1]
    return(list(pred.plm=y_hat,pred.knn=M[,1],coef=beta))
    
  } else {
    
    if(cl>1) {
      registerDoParallel(cl)
      M.new=foreach(i=1:nrow(W.new),.combine=rbind) %dopar% {knn.mean(i,W.new)}
      stopImplicitCluster()
    } else {
      M.new=foreach(i=1:nrow(W.new),.combine=rbind) %do% {knn.mean(i,W.new)}
    }
    rownames(M.new)=NULL
    
    y_hat=as.vector((X.new-M.new[,-1])%*%beta)+M.new[,1]
    return(list(pred.plm=y_hat,pred.knn=M.new[,1],coef=beta))
    
  }
  
}


pfm=function(y,pred) {
  e2=(y-pred)^2
  R2=1-sum(e2)/sum((y-mean(y))^2)
  RMSE=sqrt(mean(e2))
  return(c(R2=R2,RMSE=RMSE))
}


plm.knn.tune=function(k.grid,lambda.grid=1,y,X,W,fold=4,metric="R2",cl=1) {
  
  n=length(y)
  group=caret::createFolds(1:n,k=fold)
  
  plm.knn.cv=function(k,lambda) {
    y_hat=rep(NA,n)
    for (i in 1:fold) {
      y_hat[group[[i]]]=plm.knn(k=k,
                                y=y[-group[[i]]],X=X[-group[[i]],],W=W[-group[[i]],],
                                lambda=lambda,
                                X.new=X[group[[i]],],W.new=W[group[[i]],],
                                cl=cl)$pred.plm
    }
    performance=pfm(y,y_hat)
    return(performance)
  }
  
  if(length(lambda.grid)==1) {
    tuneGrid=data.frame(k=k.grid,lambda=lambda.grid)
  } else {
    tuneGrid=data.frame(k=rep(k.grid,each=nrow(lambda.grid)),lambda=lambda.grid)
  }
  
  result=NULL
  for (i in 1:nrow(tuneGrid)) {
    tc=Sys.time()
    cat(paste(i," group tuning parameters starting at: ",tc,", ",sep=""))
    result=rbind(result,
                 plm.knn.cv(k=tuneGrid$k[i],lambda=unlist(tuneGrid[i,-1])))
    tc=Sys.time()-tc;print(tc)
  }
  result=as.data.frame(result)
  
  if(metric=="R2") {
    bestTune=which.max(result$R2)
  } else {
    bestTune=which.min(result$RMSE)
  }
  
  return(list(k.best=tuneGrid$k[bestTune],lambda.best=unlist(tuneGrid[bestTune,-1]),
              Performance=cbind(tuneGrid,result)))
}


