require(fracdiff)
source("whitten-aic.R")


#load your data, n is the sample size

x<-matrix(scan("RV5min.txt", n = 2960), 2960, 1, byrow = TRUE)
x<-log(x)
x<-x-mean(x)
n<-length(x)


x<-filterx(x,n)


#select the truncation
m<-round(n^(0.7))

#the trimming proportions
trm1<-round(0.02*m)
trm2<-round(0.05*m)


freq<-seq(1,n,by=1)*(2*pi/n)
xf<-fft(x)
px<-(Re(xf)^2+Im(xf)^2)/(2*pi*n)
perdx<-px[2:n]             

#local Whittle likelihood
fn<-function(h)
{
lambda<-freq[1:m]^(2*h-1)
Gh=mean(perdx[1:m]*lambda)
Rh=log(Gh)-(2*h-1)*mean(log(freq[1:m]))
return(Rh)
}
est<-optimize(fn,c(0,1.5),tol = 0.00001) 
hhat<-est$minimum

lambda_hat<-freq[1:m]^(2*hhat-1)
Ghat=mean(perdx[1:m]*lambda_hat[1:m])
stat<-rep(0,m)

#now compute the statistic
comp1<-(perdx[1:m]*lambda_hat[1:m])/Ghat
comp2<-log(freq[1:m])-mean(log(freq[1:m])) 
stat<-cumsum((comp1-1)*comp2)/sqrt(sum(comp2^2))
stat2<-cumsum((comp1-1))/sqrt(sum(comp2^2))

z1<-max(abs(stat[trm1:m]))
z2<-max(abs(stat[trm2:m]))


#plot(stat,type='l')
print (paste("the local Whittle estimate of d:", hhat-0.5))
print ("-------the values of the test statistics are:----------")
print (paste("for the trimming proportion (epsilon) =0.02: W=", z1))
print (paste("for the trimming proportion (epsilon) =0.05: W=", z2))




