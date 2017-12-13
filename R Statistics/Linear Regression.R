data(airquality)
names(airquality)
# names in R have dots
plot(Ozone~Solar.R, data=airquality)
# True in R is T and False is F, name.dot
mean.Ozone = mean(airquality$Ozone, na.rm=T)
abline(h=mean.Ozone)

model1 = lm(Ozone~Solar.R, data=airquality)

model1

abline(model1, col='red')
plot(model1)

termplot(model1)
summary(model1)
