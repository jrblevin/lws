filterx<-function(x,n)
{
est11<-fracdiff(x, nar = 1, nma = 1, ar = 0.01, ma = 0.02)
bic11<-est11$log.likelihood*(-2)+2*2
opt_bic<-bic11
armac<-c(est11$ar,est11$ma)
opt_coef<-c(est11$ar,est11$ma,est11$d)

est10<-fracdiff(x, nar = 1, nma = 0, ar = 0)
bic10<-est10$log.likelihood*(-2)+2*1

if (bic10<opt_bic)
{armac<-c(est10$ar,0)
opt_bic<-bic10
opt_coef<-c(est10$ar,0,est10$d)
}
est01<-fracdiff(x, nar = 0, nma = 1, ma = 0)
bic01<-est01$log.likelihood*(-2)+2*1

if (bic01<opt_bic)
{armac<-c(0,est01$ma)
opt_bic<-bic01
opt_coef<-c(0,est01$ma,est01$d)
}
est00<-fracdiff(x, nar = 0, nma = 0)
bic00<-est00$log.likelihood*(-2)+2*0

if (bic00<opt_bic)
{armac<-c(0,0)
opt_bic<-bic00
opt_coef<-c(0,0,est00$d)
}

if (abs(armac[1])>=0.99)
{armac[1]=0
armac[2]=0
opt_coef<-c(0,0,est00$d)
}

if (abs(armac[2])>=0.99)
{armac[1]=0
armac[2]=0
opt_coef<-c(0,0,est00$d)
}

cvec<-coefs(armac[1],armac[2],n)


newx<-x
for ( i in 1:n)
newx[i]=t(rev(x[1:i]))%*%cvec[1:i]

return(newx)
}

coefs<-function(car,cma,n) #yt=arc*yt-1+et-mac*et-1
{
coef1<-rep(1,n)
coef1[1]<-0
coef1[3:n]<-rep(cma,(n-2))
coef1[2:n]<-cumprod(coef1[2:n]) 
coef2<-cma-car
coef<-coef1*coef2
coef[1]<-1
return(coef)
}




