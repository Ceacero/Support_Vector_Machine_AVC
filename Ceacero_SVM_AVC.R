

# Trabalho 1

# Instalando pacotes


# Ativando pacotes

library(readr)
library(caret)
library(dplyr)
library(mlr)
library(rpart)
library(e1071)


# Lendo base de dados

base_dados = read_csv2("avc.csv")


# Separando em base de treino e base de teste

inTrain = createDataPartition(base_dados$AVC,
                              p = 0.80,
                              list = FALSE)


base_treino = base_dados[inTrain,]

base_teste = base_dados[-inTrain,]



# Pre processando base_treino para treinar o modelo de impute

base_treino$gender[base_treino$gender == "Other"] = NA

base_treino$id = NULL

base_treino$gender = factor(base_treino$gender,
                            levels = c("Female", "Male"),
                            labels = c(0, 1))

base_treino$age = as.numeric(base_treino$age)

base_treino$hypertension = as.factor(base_treino$hypertension)

base_treino$heart_disease = as.factor(base_treino$heart_disease)

base_treino$ever_married = factor(base_treino$ever_married,
                                  levels = c("No", "Yes"),
                                  labels = c(0, 1))

base_treino$work_type = factor(base_treino$work_type,
                               levels = c("Private", "Self-employed",
                                          "Govt_job", "children", "Never_worked"),
                               labels = c(0, 1, 2, 3, 4))

base_treino$Residence_type = factor(base_treino$Residence_type,
                                    levels = c("Rural", "Urban"),
                                    labels = c(0, 1))

base_treino$country_birth = factor(base_treino$country_birth,
                                   levels = c("USA", "Others"),
                                   labels = c(0, 1))

base_treino$avg_glucose_level = as.numeric(base_treino$avg_glucose_level)

base_treino$bmi = as.numeric(base_treino$bmi)

base_treino$smoking_status = factor(base_treino$smoking_status,
                                    levels = c("never smoked",
                                               "formerly smoked",
                                               "smokes"),
                                    labels = c(0, 1, 2))

base_treino$AVC = as.factor(base_treino$AVC)


na_counts = sapply(base_treino, function(x) sum(is.na(x)))

na_counts_df = data.frame(Variable = names(na_counts), NA_Counts = na_counts)

ggplot(data = na_counts_df, aes(x = Variable, y = NA_Counts)) +
  geom_bar(stat = "identity",
           fill = "steelblue",
           color = "black") +
  theme_minimal() +
  labs(title = "Quantidade de NA por Variável",
       x = "Variável",
       y = "Quantidade de NA") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Treinando modelo de imputacao com base de treino

treino_imp = mlr::impute(base_treino, target = "AVC",
                         classes = list(numeric = imputeLearner("regr.rpart"),
                                        factor = imputeLearner("classif.naiveBayes")))


# Criando funcao de preprocessamento


modelo_preprocessamento = function(dados){
  
  dados$id = NULL
  
  dados$gender[dados$gender == "Other"] = NA
  
  dados$gender = factor(dados$gender,
                        levels = c("Female", "Male"),
                        labels = c(0, 1))
  
  dados$age = as.numeric(dados$age)
  
  dados$hypertension = as.factor(dados$hypertension)
  
  dados$heart_disease = as.factor(dados$heart_disease)
  
  dados$ever_married = factor(dados$ever_married,
                                    levels = c("No", "Yes"),
                                    labels = c(0, 1))
  
  dados$work_type = factor(dados$work_type,
                                 levels = c("Private", "Self-employed",
                                            "Govt_job", "children", "Never_worked"),
                                 labels = c(0, 1, 2, 3, 4))
  
  dados$Residence_type = factor(dados$Residence_type,
                                      levels = c("Rural", "Urban"),
                                      labels = c(0, 1))
  
  dados$country_birth = factor(dados$country_birth,
                                     levels = c("USA", "Others"),
                                     labels = c(0, 1))
  
  dados$avg_glucose_level = as.numeric(dados$avg_glucose_level)
  
  dados$bmi = as.numeric(dados$bmi)
  
  dados$smoking_status = factor(dados$smoking_status,
                                      levels = c("never smoked",
                                                 "formerly smoked",
                                                 "smokes"),
                                      labels = c(0, 1, 2))
  
  dados$AVC = factor(dados$AVC,
                     levels = c(0, 1),
                     labels = c(0, 1))
  
  dados = mlr::reimpute(dados, treino_imp$desc)
  
  for (i in 1:length(dados$bmi)) {
    
    if(dados$bmi[i] > 80) {
      
      dados$bmi[i] = dados$bmi[i]/10
      
    }
    
  }
  
  return(dados)
  
}

# Preprocessando base de treino

base_treino = base_dados[inTrain,]

base_treino = modelo_preprocessamento(base_treino)


ggplot(data = base_treino,
       mapping = aes(x = AVC)) +
  geom_bar(fill = "steelblue",
           color = "black") +
  ylab("Quantidade") +
  theme_bw()



# O banco de dados de treino é desbalanceado. Vamos balancear.


base_treino = upSample(x = base_treino[,1:11],
                       y = base_treino$AVC,
                       list = FALSE,
                       yname = "AVC")


# Padronizando os dados e ajustando modelo com bootstrapping com funcao train


ctrl = trainControl(method = "repeatedcv", number = 10, rep = 3)

modelFit = caret::train(AVC ~ .,
                        data = base_treino,
                        method = "svmLinear",
                        preProcess = c("BoxCox"),
                        trControl = ctrl)


base_teste = modelo_preprocessamento(base_teste)


previsao = predict(modelFit, base_teste)


metric = confusionMatrix(previsao,
                         base_teste$AVC)


as.numeric(metric$byClass["Specificity"])

as.numeric(metric$overall["Accuracy"])

as.numeric(metric$byClass["Sensitivity"])




metric_table <- as.data.frame(metric$table)
colnames(metric_table) <- c("Reference", "Prediction", "Freq")

# Criar o gráfico de calor (heatmap)
ggplot(data = metric_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "gray80", high = "olivedrab") +
  labs(title = "Matriz de Confusão",
       x = "Valor Verdadeiro",
       y = "Valor Previsto") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )






save(modelFit, treino_imp, modelo_preprocessamento,
     file = "Guilherme_Ceacero_Trab1.Rdata")




