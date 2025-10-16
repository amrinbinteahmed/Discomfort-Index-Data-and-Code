# Load Necessary Packages for Time Series
library(tidyverse)
library(readxl)
library(openxlsx)
library(forecast)
library(tseries)
library(prophet)
library(e1071)
library(randomForest)
library(xts)
library(rpart)
library(xgboost)
library(keras)
library(Metrics)
library(lmtest)
library(FinTS)
library(rugarch)

# Read the Data
df = read_xlsx("Raj_Clean_DI_Data.xlsx")
head(df)

# Check Structure and Missing Value
glimpse(df)
sum(is.na(df))

# Creating Time Series Object for Temperature
n = nrow(df) # number of observations = 15340

min_day = as.Date(df$Date[1])
max_day = as.Date(df$Date[n])

ts_df = ts(df[,4], frequency = 365.25, 
           start = c(year(min_day), as.numeric(format(min_day, "%j"))))

# Check Stationarity (Augmented Dickey-Fuller Test)
alpha = 0.05
adf.test(ts_df) # p-value = 0.01 (Stationary)

# Initial Plotting
ts.plot(ts_df)
ggseasonplot(ts_df, xlab = "Season", ylab = "Discomfort Index", main = "")
ggsubseriesplot(ts_df)
ggtsdisplay(ts_df, theme = theme_bw(base_size = 13))

# Split Data into Train and Test
train_percentage = 0.75
train_length = round(n*train_percentage) # 10958
train = df[1:train_length,]
test = df[(train_length + 1):n,]
k = nrow(test) # 3652

# Create Time Series Object for Train and Test Data and Plotting (Temperature)
ts_train = ts(df[1:train_length,4], frequency = 365.25, 
              start = c(year(min_day), as.numeric(format(min_day, "%j"))))

test_minday = as.Date(df$Date[train_length + 1])

ts_test = ts(df[(train_length + 1):n,4], frequency = 365.25, 
             start = c(year(test_minday), as.numeric(format(test_minday, "%j"))))

plot(ts_train)
plot(ts_test)


###### Fit Time Series Model in Train Data and Forecast Test Data

## 1. ARIMA
fit1 = auto.arima(ts_train, seasonal = F)
F_ARIMA = forecast::forecast(fit1, h = k)$mean

## 2. TBATS
fit2 = tbats(ts_train)
F_TBATS = forecast::forecast(fit2, h = k)$mean

## 3. ETS
fit3 = ets(ts_train)
F_ETS = forecast::forecast(fit3, h = k)$mean

## 4. GARCH

# Checking Volatility
ArchTest(ts_df, lags = 12)

# Identify the Order of ARCH
aic_values = numeric()

for (p in 1:10) {
  spec = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(p, 0)),
                    mean.model = list(armaOrder = c(0,0)))
  fit = ugarchfit(spec = spec, data = ts_df)
  aic_values[p] = infocriteria(fit)[1]
}

plot(1:10, aic_values, type = "b", xlab = "Order p", ylab = "AIC", main = "Choosing p for ARCH")

# Identify the Order of GARCH
aic_matrix = matrix(NA, nrow = 3, ncol = 3)
rownames(aic_matrix) = colnames(aic_matrix) = 1:3

for (p in 1:3) {
  for (q in 1:3) {
    spec = ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(p,q)),
      mean.model = list(armaOrder = c(0,0))
    )
    fit = ugarchfit(spec = spec, data = ts_df, solver = "hybrid")
    aic_matrix[p,q] = infocriteria(fit)[1]
  }
}

aic_matrix

# === 1. Fit ARCH(4) ===
spec_arch4 = ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(4, 0)),
  mean.model = list(armaOrder = c(0,0))
)

fit_arch4 = ugarchfit(spec = spec_arch4, data = ts_df)

# Get AIC
aic_arch4 = infocriteria(fit_arch4)[1]
cat("ARCH(4) AIC:", aic_arch4, "\n")


# === 2. Fit GARCH(1,3) ===
spec_garch13 = ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,3)),  # GARCH(1,3): (p=1, q=3)
  mean.model = list(armaOrder = c(0,0))
)

fit_garch13 = ugarchfit(spec = spec_garch13, data = ts_df)

# Get AIC
aic_garch13 = infocriteria(fit_garch13)[1]
cat("GARCH(1,3) AIC:", aic_garch13, "\n")


# === 3. Compare
if (aic_arch4 < aic_garch13) {
  cat("👉 ARCH(4) is better based on AIC.\n")
} else {
  cat("👉 GARCH(1,3) is better based on AIC.\n")
}

##### Model Fitting
spec_garch13 = ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,3)),
  mean.model = list(armaOrder = c(1,1))
)

fit4 = ugarchfit(spec = spec_garch13, data = ts_train)
forecast_garch13 = ugarchforecast(fit4, n.ahead = k)
F_GARCH = fitted(forecast_garch13)
garch_residuals = residuals(fit4)

###### Fit Machine Learning Model in Train Data and Forecast Test Data

## 5. ANN
fit5 = nnetar(ts_train)
F_ANN = forecast::forecast(fit5, h = k)$mean

## 6. Facebook Prophet
FP_train = data.frame(ds = as.Date(train$Date), y = as.matrix(ts_train))
FP_test = data.frame(ds = as.Date(test$Date), y = as.matrix(ts_test))

fit6 = prophet(FP_train)
F_FP = predict(fit6, FP_test)$yhat

fitted = predict(fit6, FP_train)$yhat
train_residuals = FP_train$y - fitted

## 7. Support Vector Regression
X1 = df$Discomfort_Index[1:train_length]
Y1 = df$Discomfort_Index[2:(train_length + 1)]
X2 = df$Discomfort_Index[train_length:(n-1)]
Y2 = df$Discomfort_Index[(train_length + 1):n]

fit7 = svm(X1, Y1, degree = 3, cost = 45.69, nu = 0.5, tolerance = 0.00001, 
           epsilon = 0.00001)

F_SVR = predict(fit7, X2)


## 8. Random Forest Regression

# Step 1 (Data Preparation)
ts = xts(df$Discomfort_Index, order.by = df$Date)
lags = 1:30
lagged_data = lag.xts(ts, lags)

lagged_df = data.frame(lagged_data)
colnames(lagged_df) = paste0("lag_", lags)

final_data = cbind(df$Date, df$Discomfort_Index, lagged_df)
colnames(final_data) = c("Date", "Discomfort Index", paste0("lag_", lags))
nrow(final_data)

# Step 2 (Split Data into Train and Test)
trainData = final_data[1:train_length,]
testData = final_data[(train_length + 1):n,]

# Step 3 (Remove Rows with Na's Created by Lagging)
train_data = trainData[complete.cases(trainData),]
test_data = testData[complete.cases(testData),]

# Step 4 (Fit a Random Forest model)
fit8 = randomForest(`Discomfort Index` ~ ., data = train_data, ntree = 100)

# Step 5 (Make Predictions on the Test Data
F_RF = predict(fit8, test_data)
rf_residuals = train_data$`Discomfort Index`- predict(fit8, train_data)

## 9. Decision Trees
fit9 = rpart(`Discomfort Index` ~ ., data = train_data, method = "anova")
F_DT = predict(fit9, test_data)
dt_residuals = train_data$`Discomfort Index` - predict(fit9, train_data)


## 10. LSTM
library(tensorflow)
library(keras)

# Data Preparation
prepare_data = function(d, n_steps){
  X = list()
  y = list()
  for(i in 1:length(d)){
    end_ix = i + n_steps - 1
    if(end_ix > length(d) - 1){
      break
    }
    seq_x = d[i:end_ix]
    seq_y = d[end_ix + 1]
    X = c(X, list(seq_x))
    y = c(y, seq_y)
  }
  return(list(X = X, y = y))
}

n_steps = 30
n_features = 1

lstm_train = df[1:(train_length - 30), 4]
lstm_test = df[((train_length - 30) + 1):n, 4]

# Define Model
model = keras_model_sequential() %>%
  layer_lstm(units = 50, activation = "relu", input_shape = c(n_steps, n_features), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = "relu") %>%
  layer_dense(units = 1)

# Compile Model
model |> compile(
  optimizer = "adam",
  loss = "mse"
)

# Define early stopping
early_stopping  = callback_early_stopping(monitor = 'val_loss', patience = 5, restore_best_weights = TRUE)

# Train-test splitting
X_train = prepare_data(lstm_train, n_steps)$X
X_train = array(as.numeric(unlist(X_train)), dim = c(length(X_train), n_steps, 1))

y_train = as.numeric(prepare_data(lstm_train, n_steps)$y)

X_test = prepare_data(lstm_test, n_steps)$X
X_test = array(as.numeric(unlist(X_test)), dim = c(length(X_test), n_steps, 1))

y_test = as.numeric(prepare_data(lstm_test, n_steps)$y)

# Fit model
h1 = model %>% fit(
  X_train, y_train,
  epochs = 10,
  batch_size = 30,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 5))
)


# Plot training history
plot(h1)

# Make predictions
F_LSTM = model |> predict(X_test)

# Residuals
train_pred = model |> predict(X_train)
lstm_res = y_train  - as.numeric(train_pred)


## 11. Build the GRU model
model_2 = keras_model_sequential() %>%
  layer_gru(units = 100, return_sequences = TRUE, input_shape = c(n_steps, 1)) %>%
  layer_gru(units = 100) %>%
  layer_dense(units = 1)

# Compile Model
model_2 |> compile(
  optimizer = "adam",
  loss = "mse"
)

# Train the model
h2 = model_2 %>% fit(
  X_train, y_train,
  epochs = 10,
  batch_size = 30,
  validation_split = 0.2,
  verbose = 2
)

# Plot training history
plot(h2)

# Make predictions
F_GRU = model_2 |> predict(X_test)


# Residuals
train_pred_2 = model_2 |> predict(X_train)
gru_residuals = y_train  - as.numeric(train_pred_2)

# 12. XG-Boost
library(xgboost)
library(caret)

prepare_data_xgb = function(d, n_steps){
  X = list()
  y = list()
  for(i in 1:(length(d) - n_steps)){
    end_ix = i + n_steps - 1
    seq_x = d[i:end_ix]
    seq_y = d[end_ix + 1]
    X = c(X, list(seq_x))
    y = c(y, seq_y)
  }
  X = do.call(rbind, X)
  y = unlist(y)
  return(list(X = X, y = y))
}

train = df$Discomfort_Index[1:(train_length - 30)]
test = df$Discomfort_Index[((train_length - 30) + 1):n]

# Prepare training set
data_train = prepare_data_xgb(train, n_steps)
X_train = data_train$X
y_train = data_train$y

# Prepare testing set
data_test = prepare_data_xgb(test, n_steps)
X_test = data_test$X
y_test = data_test$y

# === 2. Train XGBoost model ===

# Convert to xgbDMatrix
dtrain = xgb.DMatrix(data = X_train, label = y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)

# Set XGBoost parameters
params = list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,            # learning rate
  max_depth = 6,        # depth of trees
  subsample = 0.8,      # subsample ratio of training instances
  colsample_bytree = 0.8 # subsample ratio of features
)

# Train model
xgb_model = xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

# === 3. Make Predictions ===
F_XGB = predict(xgb_model, dtest)


# === 4. Residuals ===
xgb_residuals = y_train - predict(xgb_model, dtrain)


errors = data.frame(
  Model = c("ARIMA", "TBATS", "ETS", "GARCH", "ANN", "Prophet", 
            "SVR", "Random Forest", "Decision Trees", "XGBoost", "LSTM", "GRU"),
  
  MAE = c(
    mae(test_data$`Discomfort Index`, F_ARIMA),
    mae(test_data$`Discomfort Index`, F_TBATS),
    mae(test_data$`Discomfort Index`, F_ETS),
    mae(test_data$`Discomfort Index`, F_GARCH),
    mae(test_data$`Discomfort Index`, F_ANN),
    mae(test_data$`Discomfort Index`, F_FP),
    mae(test_data$`Discomfort Index`, F_SVR),
    mae(test_data$`Discomfort Index`, F_RF),
    mae(test_data$`Discomfort Index`, F_DT),
    mae(test_data$`Discomfort Index`, F_XGB),
    mae(test_data$`Discomfort Index`, F_LSTM),
    mae(test_data$`Discomfort Index`, F_GRU)
  ),
  
  MAPE = c(
    mape(test_data$`Discomfort Index`, F_ARIMA) * 100,
    mape(test_data$`Discomfort Index`, F_TBATS) * 100,
    mape(test_data$`Discomfort Index`, F_ETS) * 100,
    mape(test_data$`Discomfort Index`, F_GARCH) * 100,
    mape(test_data$`Discomfort Index`, F_ANN) * 100,
    mape(test_data$`Discomfort Index`, F_FP) * 100,
    mape(test_data$`Discomfort Index`, F_SVR) * 100,
    mape(test_data$`Discomfort Index`, F_RF) * 100,
    mape(test_data$`Discomfort Index`, F_DT) * 100,
    mape(test_data$`Discomfort Index`, F_XGB) * 100,
    mape(test_data$`Discomfort Index`, F_LSTM) * 100,
    mape(test_data$`Discomfort Index`, F_GRU) * 100
  ),
  
  RMSE = c(
    rmse(test_data$`Discomfort Index`, F_ARIMA),
    rmse(test_data$`Discomfort Index`, F_TBATS),
    rmse(test_data$`Discomfort Index`, F_ETS),
    rmse(test_data$`Discomfort Index`, F_GARCH),
    rmse(test_data$`Discomfort Index`, F_ANN),
    rmse(test_data$`Discomfort Index`, F_FP),
    rmse(test_data$`Discomfort Index`, F_SVR),
    rmse(test_data$`Discomfort Index`, F_RF),
    rmse(test_data$`Discomfort Index`, F_DT),
    rmse(test_data$`Discomfort Index`, F_XGB),
    rmse(test_data$`Discomfort Index`, F_LSTM),
    rmse(test_data$`Discomfort Index`, F_GRU)
  ),
  
  MASE = c(
    mase(test_data$`Discomfort Index`, F_ARIMA),
    mase(test_data$`Discomfort Index`, F_TBATS),
    mase(test_data$`Discomfort Index`, F_ETS),
    mase(test_data$`Discomfort Index`, F_GARCH),
    mase(test_data$`Discomfort Index`, F_ANN),
    mase(test_data$`Discomfort Index`, F_FP),
    mase(test_data$`Discomfort Index`, F_SVR),
    mase(test_data$`Discomfort Index`, F_RF),
    mase(test_data$`Discomfort Index`, F_DT),
    mase(test_data$`Discomfort Index`, F_XGB),
    mase(test_data$`Discomfort Index`, F_LSTM),
    mase(test_data$`Discomfort Index`, F_GRU)
  )
)

print(errors)
