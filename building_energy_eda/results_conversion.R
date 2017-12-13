results_long <- reshape::melt(as.data.frame(results), id = c('season', 'building'))
results_long[which(results_long$variable == "linear_r_squared"), "variable"] = "linear_rsquared"
results_long[which(results_long$variable == "random_forest_r_squared"), "variable"] = "rf_rsquared"
results_long[which(results_long$variable == "svr_r_squared"), "variable"] = "svr_rsquared"
results_long[which(results_long$variable == "random_forest_rmse"), "variable"] = "rf_rmse"
results_long[which(results_long$variable == "random_forest_mape"), "variable"] = "rf_mape"
results_long$model <- stringr::str_split_fixed(results_long$variable, "_", 2)[, 2]
results_long$model <- stringr::str_split_fixed(results_long$variable, "_", 2)[, 1]
results_long$metric <- stringr::str_split_fixed(results_long$variable, "_", 2)[, 2]
  
ggplot(filter(results_long, metric == 'rmse'), aes(season)) + 
  geom_bar(aes(y = value, fill = model), width = 0.8, position = 'dodge', stat = 'identity') + 
  facet_wrap(~building) + xlab("season") + ylab("rmse") + 
  theme(axis.text.x = element_text(angle = 90)) + 
  ggtitle("Model RMSE Comparison")

ggplot(filter(results_long, metric == 'rsquared'), aes(season)) + 
  geom_bar(aes(y = value, fill = model), width = 0.8, position = 'dodge', stat = 'identity') + 
  facet_wrap(~building) + xlab("season") + ylab("rmse") + 
  theme(axis.text.x = element_text(angle = 90)) + 
  ggtitle("Model Rsquared Comparison")

ggplot(filter(results_long, metric == 'mape'), aes(season)) + 
  geom_bar(aes(y = value, fill = model), width = 0.8, position = 'dodge', stat = 'identity') + 
  facet_wrap(~building) + xlab("season") + ylab("rmse") + 
  theme(axis.text.x = element_text(angle = 90)) + 
  ggtitle("Model Mean Average Percentage Error Comparison")
