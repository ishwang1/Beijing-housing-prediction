plm.localpoly.pred=function(y,X,W,X.new,W.new,h,p=1,kernel.fun="Epanechnikov",cl=1) {
  
  library(foreach)
  
  n=length(y)
  
  if(kernel.fun=="Epanechnikov") {
    K=function(u) {3/4*(1-u^2)*(abs(u)<=1)}
  }
  
  localpoly.mean=function(w,G) {
    D=W-matrix(rep(w,each=n),nrow=n)
    omega=apply(Matrix::Matrix(K(t(D)/h),sparse=T),MARGIN=2,FUN=prod)
    local.ind=which(omega!=0)
    Z=rep(1,length(local.ind))
    if(p>0) {
      Z=cbind(Z,poly(D[local.ind,,drop=F],degree=p,raw=T,simple=T))
    }
    ZW=t(Z*omega[local.ind])
    Proj=(solve(ZW%*%Z)%*%ZW)[1,]
    m=colSums(G[local.ind,,drop=F]*Proj)
    return(m)
  }
  
  if(cl>1) {
    doParallel::registerDoParallel(cl)
    M=foreach(i=1:n,.combine=rbind) %dopar% {localpoly.mean(w=W[i,],G=cbind(y,X))}
    doParallel::stopImplicitCluster()
  } else {
    M=foreach(i=1:n,.combine=rbind) %do% {localpoly.mean(w=W[i,],G=cbind(y,X))}
  }
  
  y_tilde=y-M[,1];X_tilde=X-M[,-1]
  beta_hat=as.vector(solve(t(X_tilde)%*%X_tilde)%*%t(X_tilde)%*%y_tilde)
  gamma_in=y-X%*%beta_hat
  
  if(cl>1) {
    doParallel::registerDoParallel(cl)
    gamma_out=foreach(i=1:nrow(W.new),.combine=c) %dopar% {localpoly.mean(w=W.new[i,],G=gamma_in)}
    doParallel::stopImplicitCluster()
  } else {
    gamma_out=foreach(i=1:nrow(W.new),.combine=c) %do% {localpoly.mean(w=W.new[i,],G=gamma_in)}
  }
  
  y_out=as.vector(X.new%*%beta_hat)+gamma_out
  return(y_out)
}


plm.localpoly.tune.cv=function(y,X,W,h.grid,p.grid,fold=4,cl=1) {
  
  n=length(y)
  tuneGrid=data.frame(p=rep(p.grid,each=nrow(h.grid)),h=h.grid)
  group=caret::createFolds(1:n,k=fold)
  
  plm.localpoly.cv=function(h,p) {
    y_hat=rep(NA,n)
    for (k in 1:fold) {
      y_hat[group[[k]]]=plm.localpoly.pred(y=y[-group[[k]]],X=X[-group[[k]],],W=W[-group[[k]],],
                                           X.new=X[group[[k]],],W.new=W[group[[k]],],h=h,p=p,cl=cl)
    }
    e2=(y-y_hat)^2
    R2=1-sum(e2)/sum((y-mean(y))^2)
    RMSE=sqrt(mean(e2))
    return(c(R2=R2,RMSE=RMSE))
  }
  
  result=NULL
  for (i in 1:nrow(tuneGrid)) {
    cat(paste("Group",i,"starting tuning at",Sys.time()),"\n")
    result=rbind(result,plm.localpoly.cv(h=unlist(tuneGrid[i,-1]),p=tuneGrid$p[i]))
  }
  result=cbind(tuneGrid,result)
  
  bestTune=which.max(result$R2)
  h.best=unlist(tuneGrid[bestTune,-1]);names(h.best)=NULL
  p.best=tuneGrid$p[bestTune]
  
  return(list(h.best=h.best,p.best=p.best,Performance=result))
}


plm.localpoly.tune.oos=function(y,X,W,h.grid,p.grid,valid.ind,cl=1) {
  
  tuneGrid=data.frame(p=rep(p.grid,each=nrow(h.grid)),h=h.grid)
  
  plm.localpoly.oos=function(h,p) {
    y_hat=plm.localpoly.pred(y=y[-valid.ind],X=X[-valid.ind,],W=W[-valid.ind,],
                             X.new=X[valid.ind,],W.new=W[valid.ind,],h=h,p=p,cl=cl)
    e2=(y[valid.ind]-y_hat)^2
    R2=1-sum(e2)/sum((y[valid.ind]-mean(y[valid.ind]))^2)
    RMSE=sqrt(mean(e2))
    return(c(R2=R2,RMSE=RMSE))
  }
  
  result=NULL
  for (i in 1:nrow(tuneGrid)) {
    cat(paste("Group",i,"starting tuning at",Sys.time()),"\n")
    result=rbind(result,plm.localpoly.oos(h=unlist(tuneGrid[i,-1]),p=tuneGrid$p[i]))
  }
  result=cbind(tuneGrid,result)
  
  bestTune=which.max(result$R2)
  h.best=unlist(tuneGrid[bestTune,-1]);names(h.best)=NULL
  p.best=tuneGrid$p[bestTune]
  
  return(list(h.best=h.best,p.best=p.best,Performance=result))
}


