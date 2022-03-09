library("dplyr")
library("ggplot2")
# STUDY 1 WITH AND WITHOUT
data = read.table("../tl/resW-15-iv.csv", sep=";", head=T)
data = data[-1]
names(data) =  c("cluster", "buidlingID", "mae", "mse", "rmse", "cvrmse", "mape")

withd = data

data = read.table("../tl/resWO-15-iv.csv", sep=";", head=T)
data = data[-1]
names(data) =  c("cluster", "buidlingID", "mae", "mse", "rmse", "cvrmse", "mape")

without = data

meanW = summarise(group_by(withd,cluster),
                  mae = mean(mae),
                  mse = mean(mse),
                  rmse = mean(rmse),
                  cvrmse = mean(cvrmse),
                  mape = mean(mape)
)

meanWO = summarise(group_by(without,cluster),
                   mae = mean(mae),
                   mse = mean(mse),
                   rmse = mean(rmse),
                   cvrmse = mean(cvrmse),
                   mape = mean(mape)
)


CVRMSE = c(meanW$cvrmse, meanWO$cvrmse)
TL = as.factor(c(rep("Yes",nrow(meanW)), rep("No",nrow(meanWO))))
cluster = as.factor(c(1:15,1:15))
df3 <- data.frame(TL, cluster, CVRMSE)


dev.off()
svg("comparison1.svg", width = 6, height = 3)
p <- ggplot(data=df3, aes(x=cluster, y=CVRMSE, fill=TL)) +
  geom_bar(stat="identity", color="black", position=position_dodge())+ theme(text = element_text(size=16),  axis.text.x = element_text(size=10))
p
dev.off()


# STUDY 1 WITH AND WITHOUT - imageclus
data = read.table("../tl/resW-15-iIMAGECLUS.csv", sep=";", head=T)
data = data[-1]
names(data) =  c("cluster", "buidlingID", "mae", "mse", "rmse", "cvrmse", "mape")

withd = data

data = read.table("../tl/resWO-15-iIMAGECLUS.csv", sep=";", head=T)
data = data[-1]
names(data) =  c("cluster", "buidlingID", "mae", "mse", "rmse", "cvrmse", "mape")

without = data

library("dplyr")
meanW = summarise(group_by(withd,cluster),
                  mae = mean(mae),
                  mse = mean(mse),
                  rmse = mean(rmse),
                  cvrmse = mean(cvrmse),
                  mape = mean(mape)
)

meanWO = summarise(group_by(without,cluster),
                   mae = mean(mae),
                   mse = mean(mse),
                   rmse = mean(rmse),
                   cvrmse = mean(cvrmse),
                   mape = mean(mape)
)


CVRMSE = c(meanW$cvrmse, meanWO$cvrmse)
TL = as.factor(c(rep("Yes",nrow(meanW)), rep("No",nrow(meanWO))))
cluster = as.factor(c(1:15,1:15))
df3 <- data.frame(TL, cluster, CVRMSE)

dev.off()
svg("comparison2.svg", width = 6, height = 3)
p <- ggplot(data=df3, aes(x=cluster, y=CVRMSE, fill=TL)) +
  geom_bar(stat="identity", color="black", position=position_dodge())+ theme(text = element_text(size=16),  axis.text.x = element_text(size=10))
p
dev.off()

