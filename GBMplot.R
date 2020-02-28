library(caret)
library(doParallel)

load("Beijing Housing/lianjia.RData")
load("Beijing Housing/result.RData")


# predicting grid

lianjia.grid=expand.grid(square=mean(lianjia$square),
                         livingRoom=round(mean(lianjia$livingRoom)),
                         drawingRoom=round(mean(lianjia$drawingRoom)),
                         kitchen=round(mean(lianjia$kitchen)),
                         bathRoom=round(mean(lianjia$bathRoom)),
                         floor_type=unique(lianjia$floor_type),
                         floor_total=round(mean(lianjia$floor_total)),
                         elevator=1,ladderRatio=mean(lianjia$ladderRatio),
                         renovationCondition=unique(lianjia$renovationCondition),
                         buildingType=unique(lianjia$buildingType),
                         buildingStructure=unique(lianjia$buildingStructure),
                         age=round(mean(lianjia$age)),fiveYearsProperty=1,
                         subway=1,district=unique(lianjia$district),
                         communityAverage=mean(lianjia$communityAverage),
                         DOM=round(mean(lianjia$DOM)),followers=round(mean(lianjia$followers)),
                         t_trade=2404)
lianjia.grid=cbind(subset(lianjia.grid,
                          floor_type=="middle" & renovationCondition=="hardcover" & 
                            buildingType=="plate" & buildingStructure=="steel-concrete composite" & district=="7"),
                   expand.grid(Lng=seq(from=116.08,to=116.71,by=0.01),
                               Lat=seq(from=39.63,to=40.25,by=0.01)))

pred.GBM=predict(result$GBM.model,newdata=lianjia.grid)
pred.GBM=cbind(lianjia.grid[,c("Lng","Lat")],price=pred.GBM)


# plot

library(ggplot2)
library(OpenStreetMap)

bjmap=openmap(upperLeft=c(max(pred.GBM$Lat),min(pred.GBM$Lng)),
              lowerRight=c(min(pred.GBM$Lat),max(pred.GBM$Lng)),
              type="esri-topo")

windows(16,12)
autoplot(openproj(bjmap))+
  geom_tile(data=pred.GBM,aes(x=Lng,y=Lat,fill=price),alpha=0.8)+scale_fill_distiller(palette="Spectral")+
  labs(x="Longitude",y="Latitude",fill=expression(paste("Price (",CNY/m^2,")",sep="")))


