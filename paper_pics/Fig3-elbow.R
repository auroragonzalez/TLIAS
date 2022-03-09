library("ggplot2")
cost = read.table("../01.k-prot-clustering/cost.txt")
cost$clusters = 1:nrow(cost)
cost$clusters = as.factor(cost$clusters)
names(cost)[1] = "cost"


ggplot(data=cost, aes(x=clusters, y=cost, group=1, color="darkred")) +
  geom_line()+
  geom_point()

dev.off()
svg("elbow.svg",width = 5, height = 3)
ggplot(data=cost, aes(x=clusters, y=cost, group=1, color="darkred")) +
  geom_line()+
  geom_point()
dev.off()
