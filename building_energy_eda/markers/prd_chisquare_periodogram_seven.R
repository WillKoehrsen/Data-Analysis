chisq.pd <- function(x, min.period, max.period, alpha) {
  N <- length(x)
  variances = NULL
  periods = seq(min.period, max.period)
  rowlist = NULL
  for(lc in periods){
    ncol = lc
    nrow = floor(N/ncol)
    rowlist = c(rowlist, nrow)
    x.trunc = x[1:(ncol*nrow)]
    x.reshape = t(array(x.trunc, c(ncol, nrow)))
    variances = c(variances, var(colMeans(x.reshape)))
  }
 # Qp = (rowlist * periods * variances) / var(x)
  
  Qp = (rowlist * (periods-1) * variances) / ((N-1)*var(x)/N)
  df = periods - 1
  pvals = 1-pchisq(Qp, df)
  pass.periods = periods[pvals<alpha]
  pass.pvals = pvals[pvals<alpha]
 # return(cbind(periods[pvals==min(pvals)], pvals[pvals==min(pvals)]))
  length_out<-length(periods[pvals==min(pvals)])
  out <- matrix(NA,length_out,2)
  colnames(out) <-c("Period","p.value")
  out[,"Period"]<-periods[pvals==min(pvals)]
  out[,"p.value"]<-pvals[pvals==min(pvals)]
  return(structure(out))
}

#energy_period<-chisq.pd(A$forecast, 2, 100, .0001)
#maxenergy_period<-energy_period[energy_period[,1]==max(energy_period[,1]),]



library(data.table)
setwd("H:/git/17-edifes/data/forecasted")

temp<-list.files(pattern = "*imp.csv")

for(z in 1:length(temp)){
  A<-fread(temp[z])
  A<-as.data.frame(A)
  
  energy_period<-chisq.pd(A$forecast, 2, 100, .0001)
  
  print(list( temp[z],energy_period))
 
  
}

plot(A$forecast[1:2000],type="l",xlab="Time",ylab="Energy consumption(kwh)"
     ,main= bquote(paste("period=", .(energy_period[1,1]), ",p-value=",.(energy_period[1,2]))))
    
                        #  , "/","period=", .(energy_period[2,1]), ",p-value=",.(energy_period[2,2]))))

  
  plot(A$PreviousRawValue[1:5000],type="l")
  plot(A$PreviousRawValue[500:1500],type="l")
  
  plot(A$PresentRawValue[1:2000],type="l",xlab="Time",ylab="Energy consumption(kwh)",main= bquote(paste("period=", .(energy_period[1,1]), ",p-value=",.(energy_period[1,2]))))
