plm.knn=function(k,y,X,W,X.new=NULL,W.new=NULL,lambda=1,p=2,cl=1) {
  
  library(foreach)
  
  n=length(y)
  if(length(lambda)==1) {lambda=rep(lambda,ncol(W))}
  
  knn.mean=function(w) {
    d=colSums((abs(t(W)-w)*lambda)^p)^(1/p)
    h=sort(d)[k]
    nb.ind=which(d<=h)
    y_bar=mean(y[nb.ind])
    x_bar=colMeans(X[nb.ind,])
    return(c(y_bar,x_bar))
  }
  
  if(cl>1) {
    doParallel::registerDoParallel(cl)
    M=foreach(i=1:n,.combine=rbind) %dopar% {knn.mean(W[i,])}
    doParallel::stopImplicitCluster()
  } else {
    M=foreach(i=1:n,.combine=rbind) %do% {knn.mean(W[i,])}
  }
  
  y_tilde=y-M[,1];X_tilde=X-M[,-1]
  beta_hat=as.vector(solve(t(X_tilde)%*%X_tilde)%*%t(X_tilde)%*%y_tilde)
  names(beta_hat)=colnames(X)
  
  y.in=as.vector(X_tilde%*%beta_hat)+M[,1]
  alpha.in=M[,1]-as.vector(M[,-1]%*%beta_hat)
  
  if(is.null(X.new)|is.null(W.new)) {
    return(list(pred.in=y.in,local.in=alpha.in,global=beta_hat))
  } else {
    if(cl>1) {
      doParallel::registerDoParallel(cl)
      M.new=foreach(i=1:nrow(W.new),.combine=rbind) %dopar% {knn.mean(W.new[i,])}
      doParallel::stopImplicitCluster()
    } else {
      M.new=foreach(i=1:nrow(W.new),.combine=rbind) %do% {knn.mean(W.new[i,])}
    }
    y.out=as.vector((X.new-M.new[,-1])%*%beta_hat)+M.new[,1]
    alpha.out=M.new[,1]-as.vector(M.new[,-1]%*%beta_hat)
    return(list(pred.out=y.out,pred.in=y.in,local.out=alpha.out,local.in=alpha.in,global=beta_hat))
  }
}


plm.knn.tune.cv=function(y,X,W,k.grid,lambda.grid=1,p=2,fold=4,cl=1) {
  
  n=length(y)
  group=caret::createFolds(1:n,k=fold)
  
  plm.knn.cv=function(k,lambda) {
    y_hat=rep(NA,n)
    for (i in 1:fold) {
      y_hat[group[[i]]]=plm.knn(k=k,
                                y=y[-group[[i]]],X=X[-group[[i]],],W=W[-group[[i]],],
                                X.new=X[group[[i]],],W.new=W[group[[i]],],
                                lambda=lambda,p=p,cl=cl)$pred.out
    }
    e2=(y-y_hat)^2
    R2=1-sum(e2)/sum((y-mean(y))^2)
    RMSE=sqrt(mean(e2))
    return(c(R2=R2,RMSE=RMSE))
  }
  
  if(length(lambda.grid)==1) {
    tuneGrid=data.frame(k=k.grid,lambda=lambda.grid)
  } else {
    tuneGrid=data.frame(k=rep(k.grid,each=nrow(lambda.grid)),lambda=lambda.grid)
  }
  
  result=NULL
  for (i in 1:nrow(tuneGrid)) {
    cat(paste("Group",i,"starting tuning at",Sys.time()),"\n")
    result=rbind(result,
                 plm.knn.cv(k=tuneGrid$k[i],lambda=unlist(tuneGrid[i,-1])))
  }
  
  result=cbind(tuneGrid,result)
  bestTune=which.max(result$R2)
  k.best=tuneGrid$k[bestTune]
  lambda.best=unlist(tuneGrid[bestTune,-1]);names(lambda.best)=NULL
  
  return(list(k.best=k.best,lambda.best=lambda.best,Performance=result))
}


plm.knn.tune.oos=function(y,X,W,valid.ind,k.grid,lambda.grid=1,p=2,cl=1) {
  
  n=length(y)
  
  plm.knn.oos=function(k,lambda) {
    y_hat=plm.knn(k=k,
                  y=y[-valid.ind],X=X[-valid.ind,],W=W[-valid.ind,],
                  X.new=X[valid.ind,],W.new=W[valid.ind,],
                  lambda=lambda,p=p,cl=cl)$pred.out
    e2=(y[valid.ind]-y_hat)^2
    R2=1-sum(e2)/sum((y[valid.ind]-mean(y[valid.ind]))^2)
    RMSE=sqrt(mean(e2))
    return(c(R2=R2,RMSE=RMSE))
  }
  
  if(length(lambda.grid)==1) {
    tuneGrid=data.frame(k=k.grid,lambda=lambda.grid)
  } else {
    tuneGrid=data.frame(k=rep(k.grid,each=nrow(lambda.grid)),lambda=lambda.grid)
  }
  
  result=NULL
  for (i in 1:nrow(tuneGrid)) {
    cat(paste("Group",i,"starting tuning at",Sys.time()),"\n")
    result=rbind(result,
                 plm.knn.oos(k=tuneGrid$k[i],lambda=unlist(tuneGrid[i,-1])))
  }
  
  result=cbind(tuneGrid,result)
  bestTune=which.max(result$R2)
  k.best=tuneGrid$k[bestTune]
  lambda.best=unlist(tuneGrid[bestTune,-1]);names(lambda.best)=NULL
  
  return(list(k.best=k.best,lambda.best=lambda.best,Performance=result))
}


