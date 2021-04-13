plm.NW=function(h,y,X,W,X.new=NULL,W.new=NULL,kernel.fun="Epanechnikov",cl=1) {
  
  library(foreach)
  
  n=length(y)
  
  if(kernel.fun=="Epanechnikov") {
    K=function(u) {3/4*(1-u^2)*(abs(u)<=1)}
  }
  
  NW.mean=function(w) {
    omega=apply(Matrix::Matrix(K((t(W)-w)/h),sparse=T),MARGIN=2,FUN=prod)
    local.ind=which(omega!=0)
    omega=omega[local.ind]
    omega=omega/sum(omega)
    y_bar=sum(omega*y[local.ind])
    x_bar=colSums(omega*X[local.ind,,drop=F])
    return(c(y_bar,x_bar))
  }
  
  if(cl>1) {
    doParallel::registerDoParallel(cl)
    M=foreach(i=1:n,.combine=rbind) %dopar% {NW.mean(W[i,])}
    doParallel::stopImplicitCluster()
  } else {
    M=foreach(i=1:n,.combine=rbind) %do% {NW.mean(W[i,])}
  }
  
  y_tilde=y-M[,1];X_tilde=X-M[,-1]
  beta_hat=as.vector(solve(t(X_tilde)%*%X_tilde)%*%t(X_tilde)%*%y_tilde)
  names(beta_hat)=colnames(X)
  
  y.in=as.vector(X_tilde%*%beta_hat)+M[,1]
  alpha.in=M[,1]-as.vector(M[,-1]%*%beta_hat)
  
  if(is.null(X.new)|is.null(W.new)) {
    return(list(pred.in=y.in,local.in=alpha.in,global=beta.hat))
  } else {
    if(cl>1) {
      doParallel::registerDoParallel(cl)
      M.new=foreach(i=1:nrow(W.new),.combine=rbind) %dopar% {NW.mean(W.new[i,])}
      doParallel::stopImplicitCluster()
    } else {
      M.new=foreach(i=1:nrow(W.new),.combine=rbind) %do% {NW.mean(W.new[i,])}
    }
    y.out=as.vector((X.new-M.new[,-1])%*%beta_hat)+M.new[,1]
    alpha.out=M.new[,1]-as.vector(M.new[,-1]%*%beta_hat)
    return(list(pred.out=y.out,pred.in=y.in,local.out=alpha.out,local.in=alpha.in,global=beta_hat))
  }
}


plm.NW.tune.cv=function(y,X,W,h.grid,fold=4,cl=1) {
  
  n=length(y)
  group=caret::createFolds(1:n,k=fold)
  
  plm.NW.cv=function(h) {
    y_hat=rep(NA,n)
    for (k in 1:fold) {
      y_hat[group[[k]]]=plm.NW(h=h,y=y[-group[[k]]],X=X[-group[[k]],],W=W[-group[[k]],],
                               X.new=X[group[[k]],],W.new=W[group[[k]],],cl=cl)$pred.out
    }
    e2=(y-y_hat)^2
    R2=1-sum(e2)/sum((y-mean(y))^2)
    RMSE=sqrt(mean(e2))
    return(c(R2=R2,RMSE=RMSE))
  }
  
  result=NULL
  for (i in 1:nrow(h.grid)) {
    cat(paste("Group",i,"starting tuning at",Sys.time()),"\n")
    result=rbind(result,plm.NW.cv(h=unlist(h.grid[i,])))
  }
  
  result=cbind(h.grid,result)
  bestTune=which.max(result$R2)
  h.best=unlist(h.grid[bestTune,]);names(h.best)=NULL
  
  return(list(h.best=h.best,Performance=result))
}


plm.NW.tune.oos=function(y,X,W,h.grid,valid.ind,cl=1) {
  
  plm.NW.oos=function(h) {
    y_hat=plm.NW(h=h,y=y[-valid.ind],X=X[-valid.ind,],W=W[-valid.ind,],
                 X.new=X[valid.ind,],W.new=W[valid.ind,],cl=cl)$pred.out
    e2=(y[valid.ind]-y_hat)^2
    R2=1-sum(e2)/sum((y[valid.ind]-mean(y[valid.ind]))^2)
    RMSE=sqrt(mean(e2))
    return(c(R2=R2,RMSE=RMSE))
  }
  
  result=NULL
  for (i in 1:nrow(h.grid)) {
    cat(paste("Group",i,"starting tuning at",Sys.time()),"\n")
    result=rbind(result,plm.NW.oos(h=unlist(h.grid[i,])))
  }
  
  result=cbind(h.grid,result)
  bestTune=which.max(result$R2)
  h.best=unlist(h.grid[bestTune,]);names(h.best)=NULL
  
  return(list(h.best=h.best,Performance=result))
}


