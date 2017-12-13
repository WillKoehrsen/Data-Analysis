

#Progressive Buildings 

library(data.table)
library(bcpa)
library(ggplot2)

setwd("~/17-edifes/data/Progressive")

temp<-list.files(pattern = "f-*")


for(z in 1:length(temp)){
  
  #for(z in 1:1){
  
  A<-fread(temp[z])
  A<-as.data.frame(A)
  A$timestamp<-as.POSIXct(A$timestamp)
  A$forecast<-A$elec_cons
  
  
  t <- 1:length(A$forecast)
  bestbreak <- GetBestBreak(A$forecast,t,range=1, tau=FALSE)
  
  YearsNumber<-ceiling(dim(A)[1]/(96*365))
  
  changetime<-A$timestamp[bestbreak[2]]
  
  print(ggplot(A,aes(x=timestamp,y=elec_cons))+geom_line()+
          labs(title = "Consumption")+
          labs(y= "KwH", x="Date")+
          theme(axis.title = element_text(size = 16,face= 'bold')) +
          theme(plot.title = element_text(size = 16,face= 'bold')) +
          theme(strip.text = element_text(size = 12,   face = "bold")) +
          theme(legend.text=element_text(size=12))+
          theme(axis.text  = element_text(size = 12, face = 'bold'))+
          theme(legend.title=element_blank())+ theme(legend.position = c(0.95, .95))+
          geom_vline(xintercept=changetime,color="red",size=2))
  
  
  #  plot(t,A$forecast,type="l",main=c("Complete Data of ",temp[z], YearsNumber,"Years"),ylab="Energy",xlab="time")
  #abline(v = bestbreak[2], col="red")
  
  
  
  TChangeMonth<-as.integer(format(as.Date(changetime, format="%Y/%m/%d"),"%m"))
  
  print(list( "Building"=temp[z],"Total Change time"=changetime))
  
  ChangeMonth<-NULL
  if(dim(A)[1]>96*365){
    
    A$timestamp<-as.POSIXct(A$timestamp)
    A$time<-as.Date(as.character(A$timestamp))
    
    print(list("MinTime"= min(A$timestamp),"MaxTime"= max(A$timestamp)))
    
    
    
    yearsubset<-NULL
    yearsubset<-A[1:(96*365),]
    
    
    t <- 1:length(yearsubset$forecast)
    bestbreak <- GetBestBreak(yearsubset$forecast,t,range=1, tau=FALSE)
    #plot(t,yearsubset$forecast,type="l",main="Year 1",ylab="Energy",xlab="time")
    #abline(v = bestbreak[2], col="red")
    changetime<-yearsubset$timestamp[bestbreak[2]]
    
    
    
    print(ggplot(yearsubset,aes(x=timestamp,y=elec_cons))+geom_line()+
            labs(title = "Year 1")+
            labs(y= "KwH", x="Date")+
            theme(axis.title = element_text(size = 16,face= 'bold')) +
            theme(plot.title = element_text(size = 16,face= 'bold')) +
            theme(strip.text = element_text(size = 12,   face = "bold")) +
            theme(legend.text=element_text(size=12))+
            theme(axis.text  = element_text(size = 12, face = 'bold'))+
            theme(legend.title=element_blank())+ theme(legend.position = c(0.95, .95))+
            geom_vline(xintercept=changetime,color="red",size=2))
    
    
    ChangeMonth[1]<-as.integer(format(as.Date(changetime, format="%Y/%m/%d"),"%m"))
    
    
    print(list("Year"= 1,"Change time"=changetime))
    
    
    
    for(i in 1:{YearsNumber-1}){
      
      if(i<=YearsNumber-2)
      {yearsubset<-A[{96*365*i+1}:{96*365*(i+1)},]
      }else{
        yearsubset<-A[{96*365*i+1}:{dim(A)[1]},] 
        print(list("Number of last year days"=dim(yearsubset)[1]/96))
      }
      
      t <- 1:length(yearsubset$forecast)
      bestbreak <- GetBestBreak(yearsubset$forecast,t,range=1, tau=FALSE)
      #plot(t,yearsubset$forecast,type="l", main= c("Year",i+1),ylab="Energy",xlab="time")
      # abline(v = bestbreak[2], col="red")
      changetime<-yearsubset$timestamp[bestbreak[2]]
      
      
      print(ggplot(yearsubset,aes(x=timestamp,y=elec_cons))+geom_line()+
              labs(title= i+1)+
              labs(y= "KwH", x="Date")+
              theme(axis.title = element_text(size = 16,face= 'bold')) +
              theme(plot.title = element_text(size = 16,face= 'bold')) +
              theme(strip.text = element_text(size = 12,   face = "bold")) +
              theme(legend.text=element_text(size=12))+
              theme(axis.text  = element_text(size = 12, face = 'bold'))+
              theme(legend.title=element_blank())+ theme(legend.position = c(0.95, .95))+
              geom_vline(xintercept=changetime,color="red",size=2))
      
      
      ChangeMonth[i+1]<-as.integer(format(as.Date(changetime, format="%Y/%m/%d"),"%m"))
      
      print(list("Year"= i+1,"Change time"=changetime))
      
    }
    
    
  }
  
  
  print(list("Complete data change month"=TChangeMonth,"Change Months for different years"=ChangeMonth))
}


#KSU Buildings 


library(data.table)
library(bcpa)

setwd("~/17-edifes/data/KSU/KWH")

temp<-list.files(pattern = "f-*")


for(z in 1:length(temp)){
  
  #for(z in 1:1){ 
  A<-fread(temp[z])
  A<-as.data.frame(A)
  A$timestamp<-as.POSIXct(A$timestamp)
  A$forecast<-A$elec_cons 
  
  t <- 1:length(A$forecast)
  bestbreak <- GetBestBreak(A$forecast,t,range=1, tau=FALSE)
  
  YearsNumber<-ceiling(dim(A)[1]/(96*365))
  
  # plot(t,A$forecast,type="l",main=c("Complete Data of ",temp[z], YearsNumber,"Years"),ylab="Energy",xlab="time")
  # abline(v = bestbreak[2], col="red")
  
  changetime<-A$timestamp[bestbreak[2]]
  
  
  print(ggplot(A,aes(x=timestamp,y=elec_cons))+geom_line()+
          labs(title = "Consumption")+
          labs(y= "KwH", x="Date")+
          theme(axis.title = element_text(size = 16,face= 'bold')) +
          theme(plot.title = element_text(size = 16,face= 'bold')) +
          theme(strip.text = element_text(size = 12,   face = "bold")) +
          theme(legend.text=element_text(size=12))+
          theme(axis.text  = element_text(size = 12, face = 'bold'))+
          theme(legend.title=element_blank())+ theme(legend.position = c(0.95, .95))+
          geom_vline(xintercept=changetime,color="red",size=2))
  
  
  print(list( "Building"=temp[z],"Total Change time"=changetime))
  
  TChangeMonth<-as.integer(format(as.Date(changetime, format="%Y/%m/%d"),"%m"))
  
  ChangeMonth<-NULL
  if(dim(A)[1]>96*365){
    
    A$timestamp<-as.POSIXct(A$timestamp)
    A$time<-as.Date(as.character(A$timestamp))
    
    print(list("MinTime"= min(A$timestamp),"MaxTime"= max(A$timestamp)))
    
    
    
    yearsubset<-NULL
    yearsubset<-A[1:(96*365),]
    
    
    t <- 1:length(yearsubset$forecast)
    bestbreak <- GetBestBreak(yearsubset$forecast,t,range=1, tau=FALSE)
    # plot(t,yearsubset$forecast,type="l",main="Year 1",ylab="Energy",xlab="time")
    # abline(v = bestbreak[2], col="red")
    changetime<-yearsubset$timestamp[bestbreak[2]]
    
    
    print(ggplot(yearsubset,aes(x=timestamp,y=elec_cons))+geom_line()+
            labs(title = "Year 1")+
            labs(y= "KwH", x="Date")+
            theme(axis.title = element_text(size = 16,face= 'bold')) +
            theme(plot.title = element_text(size = 16,face= 'bold')) +
            theme(strip.text = element_text(size = 12,   face = "bold")) +
            theme(legend.text=element_text(size=12))+
            theme(axis.text  = element_text(size = 12, face = 'bold'))+
            theme(legend.title=element_blank())+ theme(legend.position = c(0.95, .95))+
            geom_vline(xintercept=changetime,color="red",size=2))
    
    
    ChangeMonth[1]<-as.integer(format(as.Date(changetime, format="%Y/%m/%d"),"%m"))
    
    
    print(list("Year"= 1,"Change time"=changetime))
    
    
    
    for(i in 1:{YearsNumber-1}){
      
      if(i<=YearsNumber-2)
      {yearsubset<-A[{96*365*i+1}:{96*365*(i+1)},]
      }else{
        yearsubset<-A[{96*365*i+1}:{dim(A)[1]},] 
        print(list("Number of last year days"=dim(yearsubset)[1]/96))
      }
      
      t <- 1:length(yearsubset$forecast)
      bestbreak <- GetBestBreak(yearsubset$forecast,t,range=1, tau=FALSE)
      # plot(t,yearsubset$forecast,type="l", main= c("Year",i+1),ylab="Energy",xlab="time")
      # abline(v = bestbreak[2], col="red")
      changetime<-yearsubset$timestamp[bestbreak[2]]
      
      print(ggplot(yearsubset,aes(x=timestamp,y=elec_cons))+geom_line()+
              labs(title = i+1)+
              labs(y= "KwH", x="Date")+
              theme(axis.title = element_text(size = 16,face= 'bold')) +
              theme(plot.title = element_text(size = 16,face= 'bold')) +
              theme(strip.text = element_text(size = 12,   face = "bold")) +
              theme(legend.text=element_text(size=12))+
              theme(axis.text  = element_text(size = 12, face = 'bold'))+
              theme(legend.title=element_blank())+ theme(legend.position = c(0.95, .95))+
              geom_vline(xintercept=changetime,color="red",size=2))
      
      
      ChangeMonth[i+1]<-as.integer(format(as.Date(changetime, format="%Y/%m/%d"),"%m"))
      
      print(list("Year"= i+1,"Change time"=changetime))
      
    }
    
  }
  print(list("Complete data change month"=TChangeMonth,"Change Month for different years"=ChangeMonth))
}


