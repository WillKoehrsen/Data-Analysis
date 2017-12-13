
setwd("~/17-edifes/data/Progressive")


library(data.table)
library(ggplot2)

data_preprocess=function(dat){
  p=which(as.integer(dat$num_time)==0)
  if(p[1]>0){
    l=p[1]-1
    dat = dat[-c(1:l),]
  }
  last<-which(as.integer(dat$num_time,2)==23)
  last<-max(last)
  if(last<length(dat$num_time)){
    full=length(dat$num_time)
    dat=dat[-c({last+1}:full),]
  }
  
  return(dat)
}

temp<-list.files(pattern = "f-*")

for(z in 1:length(temp)){
  
  A<-fread(temp[z])
  print(temp[z])
  A<-as.data.frame(A)
  A$timestamp<-as.POSIXct(A$timestamp)
  A$time<-as.Date(as.character(A$timestamp))
  
  A<-data_preprocess(A)
  A$elec_cons<-A$forecast
  
  
  A_biz<-A[which(A$biz_day==1),]
  A_nonbiz<-A[which(A$biz_day==0),]
  
  
  
  Base_biz_data<-A_biz[A_biz$num_time<8.5 | A_biz$num_time>15.5,]
  pick_biz_data<-A_biz[A_biz$num_time>=8.5 & A_biz$num_time<=15.5,]
  
  
  Base_biz_meanEnergy<-aggregate(Base_biz_data$elec_cons,by=list(Base_biz_data$time),mean)
  colnames(Base_biz_meanEnergy) <- c("Date","BaseEnergyMean")
  
  pick_biz_meanEnergy<-aggregate(pick_biz_data$elec_cons,by=list(pick_biz_data$time),mean)
  colnames(pick_biz_meanEnergy) <- c("Date","PickEnergyMean")
  
  biz_meanEnergy<-cbind(Base_biz_meanEnergy,pick_biz_meanEnergy)
  biz_meanEnergy<-biz_meanEnergy[,-3]
  
  Biz_PickBase<-t.test(biz_meanEnergy$PickEnergyMean,biz_meanEnergy$BaseEnergyMean,paired=TRUE)
  
  if(Biz_PickBase$p.value>0.01){cat("\n Mean energy consumption of Night and Day for bussines days are the same.",sep="\n")
  } else {
    cat(" Mean energy consumption of Night and Day for bussines days are different.",sep="\n")
  }
  
  if (Biz_PickBase$estimate>0){cat("Building doesn't have renewable energy.",sep="\n")
  } else {
    cat(" Building has a renewable energy",sep="\n")
  }
  
  
  plot10<-ggplot(A[1:3000,],aes(x = timestamp, y = forecast)) +
    geom_line(size = 1) +
    ylab("Energy") +
    xlab("Date") 
  
  print(list(temp[z],Biz_PickBase,plot10))
}


strftime(A$timestamp[which(A$sun_rise_set=="set")],format="%H:%M")

min(strftime(A$timestamp[which(A$sun_rise_set=="set")],format="%H:%M"))

max(strftime(A$timestamp[which(A$sun_rise_set=="rise")],format="%H:%M"))
