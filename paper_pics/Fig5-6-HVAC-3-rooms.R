cvrmse = matrix(NA,23,15)
sequencia = c(seq(1,22),24)
for(i in 1:length(sequencia)){
  TL1 <- read.table(paste0("../HVAC/resW-hvac",sequencia[i],".csv"), sep=";")
  cvrmse[i,] = as.numeric(TL1[5,-1])
}

NOTL = read.table("../HVAC/resWO-hvac24.csv", sep=";")
NOTL = NOTL[,2:24]
cvrmseNO = NOTL[6,]




df <- data.frame(1:15, cvrmse[1,], rep(as.numeric(cvrmseNO[1]),15), cvrmse[2,], rep(as.numeric(cvrmseNO[2]),15), cvrmse[3,], rep(as.numeric(cvrmseNO[3]),15))
names(df) = c("cl", "TL1", "NOTL1", "TL2", "NOTL2", "TL3", "NOTL3")
library(tidyr)
data_long <- gather(df, condition, CVRMSE, TL1:NOTL3, factor_key=TRUE)

data_long$room = c(rep(1,30),rep(2,30),rep(3,30))

library("ggplot2")

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

n = 3
cols = gg_color_hue(n)

p = ggplot(data=data_long, aes(x=cl, y=CVRMSE, group=condition)) +
  geom_line(col = c(rep(cols[1],30),rep(cols[2],30),rep(cols[3],30)))+
  geom_point(col = c(rep(cols[1],30),rep(cols[2],30),rep(cols[3],30)))
p+  scale_x_continuous("cluster", labels = as.character(data_long$cl), breaks = data_long$cl) + theme(text = element_text(size=16),  axis.text.x = element_text(size=10))

dev.off()
svg("hvac.svg", height = 4, width = 6)
p+  scale_x_continuous("cluster", labels = as.character(data_long$cl), breaks = data_long$cl) + theme(text = element_text(size=16),  axis.text.x = element_text(size=10))
dev.off()





dat = data.frame(cvrmse)
dat$room <- row.names(dat)
dat.m <- melt(dat, id.vars = "room")
ggplot(dat.m, aes(room, value)) + geom_boxplot()

boxplot(t(cvrmse))
dat$room <- row.names(cvrmse)
library(reshape2)
dat.m$room = factor(dat.m$room, levels = 1:23, ordered = TRUE)
dat.m <- melt(dat, id.vars = "room")
names(dat.m)[3] = "CVRMSE"

ref.dat =  data.frame(room = factor(seq(1:23)), CVRMSE = as.numeric(cvrmseNO))
p = ggplot(dat.m, aes(room, CVRMSE, fill=room)) + geom_boxplot() +
  theme(legend.position="none") + scale_fill_manual(values=gg_color_hue(23))

p + geom_line(data = ref.dat, aes(room,CVRMSE, group=1), color="red") 



dev.off()
svg("hvac3.svg", height = 4, width = 6)
p + geom_line(data = ref.dat, aes(room,CVRMSE, group=1), color="red") + theme(text = element_text(size=16),  axis.text.x = element_text(size=10))
dev.off()


# Average improvement
TL3 = as.numeric(cvrmse[,3])
NOTL = as.numeric(cvrmseNO)


