################################################################
#Steps in implementing a logistic regression model:
#Business Understanding
#Data Understanding
#Data Preparation & EDA
#Model Building 
#Model Evaluation
################################################################

#---------------------------Business Understanding---------------------------

# Every year, around 15% employees of XYZ Company leave the company and need to be replaced 
# with the talent pool available in the job market. The management of XYZ has contracted an HR 
# analytics firm to understand what factors they should focus on, in order to curb attrition.

## AIM:

# The aim is to automate the process of predicting 
# if an employee would leave the firm or not; and to find the factors affecting the attrition. 
# The following data sets have been provided:

# 1. General Data (Employee Information)
# 2. Employee Survey Data
# 3. Manager Survey Data
# 4. In and Out times of an employee

#---------------------------Data Understanding---------------------------

### Install and Load the required packages
#install.packages("MASS")
#install.packages("car")
#install.packages("e1071")
#install.packages("caret")
#install.packages("cowplot")
#install.packages("GGally")
#install.packages("caTools")
#install.packages("ROCR")

require(dplyr)
library(dplyr)
library(MASS)
library(car)
library(e1071)
library(caret)
library(ggplot2)
library(cowplot)
library(caTools)
library(GGally)
library(ROCR)

# Loading files
general_data<- read.csv("general_data.csv", stringsAsFactors = F)
employee_survey_data<- read.csv("employee_survey_data.csv", stringsAsFactors = F)
manager_survey_data<- read.csv("manager_survey_data.csv", stringsAsFactors = F)
in_data<- read.csv("in_time.csv", stringsAsFactors = F)
out_data<- read.csv("out_time.csv", stringsAsFactors = F)

#Understanding structure of the data 
str(general_data)    # 4410 of 24 variables including the target variable Attrition
str(employee_survey_data) # 4410 obs of 4 variables
str(manager_survey_data) # 4410 obs of 3 variables
str(in_data) #4410 obs of 262 variables (employee in time on different dates)
str(out_data) #4410 obs of 262 variables (employee out time on different dates)

#The EmployeeID is stored in column "X" in in_data and out_data

# Collate the data together in one single file
# Check unique EmployeeID in each dataset
length(unique(tolower(general_data$EmployeeID)))    # 4410, confirming EmployeeID is key 
length(unique(tolower(employee_survey_data$EmployeeID)))    # 4410, confirming EmployeeID is key 
length(unique(tolower(manager_survey_data$EmployeeID)))    # 4410, confirming EmployeeID is key 
length(unique(tolower(in_data$X)))    # 4410, confirming X is key 
length(unique(tolower(out_data$X)))    # 4410, confirming X is key 

#check if EmployeeID are same across all datasets
setdiff(general_data$EmployeeID, employee_survey_data$EmployeeID) # Identical EmployeeID across these datasets
setdiff(general_data$EmployeeID, manager_survey_data$EmployeeID) # Identical EmployeeID across these datasets
setdiff(general_data$EmployeeID, in_data$X) # Identical EmployeeID across these datasets
setdiff(general_data$EmployeeID, out_data$X) # Identical EmployeeID across these datasets

#First let's merge general_data, employee_survey_data and manager_survey_data;
#as in_data and out_data needs some preprocessing 
all_data <- merge(general_data, employee_survey_data, by="EmployeeID", all = F)
all_data <- merge(all_data, manager_survey_data, by="EmployeeID", all = F)

#Now let's retrieve meaningful data from in_data and out_data and add some new metrics
#First check if all columns are the same in both data sets
setdiff(colnames(in_data), colnames(out_data)) # Identical colnames

#Convert all dates to posixct format in in_data and out_data
temp_data <- as.data.frame(sapply(in_data[c(2:262)], function(x) { return (as.POSIXct(strptime(x, "%Y-%m-%d %H:%M:%S")))})) 
in_data <- cbind(in_data[,1], temp_data)

temp_data <- as.data.frame(sapply(out_data[c(2:262)], function(x) { return (as.POSIXct(strptime(x, "%Y-%m-%d %H:%M:%S")))})) 
out_data <- cbind(out_data[,1], temp_data)

#Find time spent by an employee in the company
temp_data <- out_data[,c(2:262)] - in_data[,c(2:262)]
all_data$AvgTime <- apply(temp_data, 1, mean, na.rm=T)
all_data$AvgTime <- all_data$AvgTime/3600 #Convert to hours

#Find number of holidays the company gives in a year - 
#Logic: if all employees have in_time = NA on a particular day, it's a holiday
is_all_na <- function(x) {
  if (length(which(is.na(x)))==4410) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

holiday_data <- apply(in_data[2:262], 2, is_all_na)
holiday_count <- length(which(holiday_data==TRUE)) #12 Holidays

#Find leaves taken by an employee in the year
all_data$leaveCount <- apply(in_data[2:262], 1, function(x) { return (length(which(is.na(x)))-holiday_count) })

str(all_data)

#Check for duplicate rows
nrow(unique(all_data[2:ncol(all_data)])) #4410 - No duplicates presents

#---------------------------Data Preparation & EDA---------------------------

#Remove irrelevant columns
#EmployeeID (column 1) can be removed as it's irrelevant to the task
#EmployeeCount (column 9) always has value=1 so it can be removed
which(all_data$EmployeeCount!=1)
#Over18 (column 16) always has value=Y so it can be removed
which(all_data$Over18!="Y")
#StandardHours (column 18) always has value=8 so it can be removed
which(all_data$StandardHours!=8)
all_data <- all_data[,-c(1,9,16,18)]

str(all_data) #4410 obs. of 27 variables;
#Numeric columns:
#Age, DistanceFromHome, MonthlyIncome, NumCompaniesWorked, 
#PercentSalaryHike, TotalWorkingYears, TrainingTimesLastYear,
#YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, AvgTime, leaveCount

#Categorical but stored as numeric:
#Education, JobLevel, StockOptionLevel, EnvironmentSatisfaction, JobSatisfaction,
#WorkLifeBalance, JobInvolvement, PerformanceRating

#Categorical columns:
#Attrition, BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus

# ---------------------------- Data Preparation ---------------------------- 

# De-Duplication
# not needed

#Missing Value Treatment
#Let's see which columns have NA values present
sapply(all_data, function(x) sum(is.na(x))) 

#Columns: NumCompaniesWorked, TotalWorkingYears, EnvironmentSatisfaction, JobSatisfaction, 
#WorkLifeBalance have missing values

#NumCompaniesWorked
#If employee has worked all his years at current company set NumCompaniesWorked to 1, else set to average of NumCompaniesWorked
all_data$NumCompaniesWorked <- ifelse(!is.na(all_data$NumCompaniesWorked), all_data$NumCompaniesWorked,
                                      ifelse(all_data$YearsAtCompany==all_data$TotalWorkingYears, 1, round(mean(all_data$NumCompaniesWorked, na.rm = T))))
sum(is.na(all_data$NumCompaniesWorked)) #Now 0

#TotalWorkingYears
#If NumCompaniesWorked=1 then TotalWorkingYears = YearsAtCompany; else TotalWorkingYears = Age-21 (usual age when people start working)
all_data$TotalWorkingYears <- ifelse(!is.na(all_data$TotalWorkingYears), all_data$TotalWorkingYears,
                                     ifelse(all_data$NumCompaniesWorked==1, all_data$YearsAtCompany, all_data$Age-21))
sum(is.na(all_data$TotalWorkingYears)) #Now 0

#EnvironmentSatisfaction
#We can set employee's missing EnvironmentSatisfaction value as average of their JobSatisfaction and WorkLifeBalance
all_data$EnvironmentSatisfaction <- ifelse(!is.na(all_data$EnvironmentSatisfaction), all_data$EnvironmentSatisfaction,
                                           round((all_data$JobSatisfaction+all_data$WorkLifeBalance)/2))
sum(is.na(all_data$EnvironmentSatisfaction))

#JobSatisfaction
#We can set employee's missing JobSatisfaction value as average of their EnvironmentSatisfaction and WorkLifeBalance
all_data$JobSatisfaction <- ifelse(!is.na(all_data$JobSatisfaction), all_data$JobSatisfaction,
                                   round((all_data$EnvironmentSatisfaction+all_data$WorkLifeBalance)/2))
sum(is.na(all_data$EnvironmentSatisfaction))

#WorkLifeBalance
#We can set employee's missing WorkLifeBalance value as average of their JobSatisfaction and EnvironmentSatisfaction
all_data$WorkLifeBalance <- ifelse(!is.na(all_data$WorkLifeBalance), all_data$WorkLifeBalance,
                                   round((all_data$JobSatisfaction+all_data$EnvironmentSatisfaction)/2))
sum(is.na(all_data$EnvironmentSatisfaction))

which(is.na(all_data)) #No more missing data

### Outlier Treatment
# Find which columns have outliers
box_theme<- theme(axis.line=element_blank(),axis.title=element_blank(), 
                  axis.ticks=element_blank(), axis.text=element_blank())

box_theme_y<- theme(axis.line.y=element_blank(),axis.title.y=element_blank(), 
                    axis.ticks.y=element_blank(), axis.text.y=element_blank(),
                    legend.position="none")

plot_grid(ggplot(all_data, aes(Age))+ geom_histogram(binwidth = 5),
          ggplot(all_data, aes(x="",y=Age))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#No outliers in Age

plot_grid(ggplot(all_data, aes(DistanceFromHome))+ geom_histogram(binwidth = 3),
          ggplot(all_data, aes(x="",y=DistanceFromHome))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#No outliers in DistanceFromHome

plot_grid(ggplot(all_data, aes(MonthlyIncome))+ geom_histogram(binwidth = 20000),
          ggplot(all_data, aes(x="",y=MonthlyIncome))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#MonthlyIncome has outliers

plot_grid(ggplot(all_data, aes(PercentSalaryHike))+ geom_histogram(binwidth = 2),
          ggplot(all_data, aes(x="",y=PercentSalaryHike))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#No outliers in PercentSalaryHike

plot_grid(ggplot(all_data, aes(TotalWorkingYears))+ geom_histogram(binwidth = 5),
          ggplot(all_data, aes(x="",y=TotalWorkingYears))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#TotalWorkingYears has some outliers

plot_grid(ggplot(all_data, aes(TrainingTimesLastYear))+ geom_histogram(binwidth = 1),
          ggplot(all_data, aes(x="",y=TrainingTimesLastYear))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#TrainingTimesLastYear has some outliers

plot_grid(ggplot(all_data, aes(YearsAtCompany))+ geom_histogram(binwidth = 5),
          ggplot(all_data, aes(x="",y=YearsAtCompany))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#YearsAtCompany has some outliers

plot_grid(ggplot(all_data, aes(YearsSinceLastPromotion))+ geom_histogram(binwidth = 2),
          ggplot(all_data, aes(x="",y=YearsSinceLastPromotion))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#YearsSinceLastPromotion has some outliers

plot_grid(ggplot(all_data, aes(YearsWithCurrManager))+ geom_histogram(binwidth = 2),
          ggplot(all_data, aes(x="",y=YearsWithCurrManager))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#YearsWithCurrManager has some outliers

plot_grid(ggplot(all_data, aes(AvgTime))+ geom_histogram(binwidth = 2),
          ggplot(all_data, aes(x="",y=AvgTime))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#AvgTime has some outliers

plot_grid(ggplot(all_data, aes(leaveCount))+ geom_histogram(binwidth = 2),
          ggplot(all_data, aes(x="",y=leaveCount))+ geom_boxplot(width=0.1)+coord_flip()+box_theme, 
          align = "v",ncol = 1)
#No outliers in leaveCount

#MonthlyIncome, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, 
#YearsSinceLastPromotion, YearsWithCurrManager, AvgTime have outliers

#MonthlyIncome
quantile(all_data$MonthlyIncome, seq(0,1,0.1))
boxplot.stats(all_data$MonthlyIncome)$out #342 outliers
min(boxplot.stats(all_data$MonthlyIncome)$out) #165950
#Values higher than or equal to 165950 are the outliers so let's set them to 165900
all_data[which(all_data$MonthlyIncome>=165950), "MonthlyIncome"] <- 165900

#TotalWorkingYears
quantile(all_data$TotalWorkingYears, seq(0,1,0.1))
boxplot.stats(all_data$TotalWorkingYears)$out #189 outliers
min(boxplot.stats(all_data$TotalWorkingYears)$out) #29
#Values higher than or equal to 29 are the outliers so let's set them to 28
all_data[which(all_data$TotalWorkingYears>=29), "TotalWorkingYears"] <- 28

#TrainingTimesLastYear
quantile(all_data$TrainingTimesLastYear, seq(0,1,0.1))
boxplot.stats(all_data$TrainingTimesLastYear)$out #714 outliers
#These are not really outliers and can be trated as normal 

#YearsAtCompany
quantile(all_data$YearsAtCompany, seq(0,1,0.1))
boxplot.stats(all_data$YearsAtCompany)$out #312 outliers
min(boxplot.stats(all_data$YearsAtCompany)$out) #19
#Values higher than or equal to 19 are the outliers so let's set them to 18
all_data[which(all_data$YearsAtCompany>=19), "YearsAtCompany"] <- 18

#YearsSinceLastPromotion
quantile(all_data$YearsSinceLastPromotion, seq(0,1,0.1))
boxplot.stats(all_data$YearsSinceLastPromotion)$out #321 outliers
min(boxplot.stats(all_data$YearsSinceLastPromotion)$out) #8
#Values higher than or equal to 8 are the outliers so let's set them to 7
all_data[which(all_data$YearsSinceLastPromotion>=8), "YearsSinceLastPromotion"] <- 7

#YearsWithCurrManager
quantile(all_data$YearsWithCurrManager, seq(0,1,0.1))
boxplot.stats(all_data$YearsWithCurrManager)$out #42 outliers
min(boxplot.stats(all_data$YearsWithCurrManager)$out) #15
#Values higher than or equal to 15 are the outliers so let's set them to 14
all_data[which(all_data$YearsWithCurrManager>=15), "YearsWithCurrManager"] <- 14

#AvgTime
quantile(all_data$AvgTime, seq(0,1,0.1))
boxplot.stats(all_data$AvgTime)$out #37 outliers
#These are not really outliers and can be trated as normal 

#Outlier treatment done

# -------------------- Exploratory Data Analysis -------------------- 

#Convert education from numeric labels to categories
all_data$Education <- ifelse(all_data$Education==1, "Below College",
                             ifelse(all_data$Education==2, "College",
                                    ifelse(all_data$Education==3, "Bachelor",
                                           ifelse(all_data$Education==4, "Master", "Doctor"))))

# Barcharts for categorical features with stacked attrition information
bar_theme1<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
                   legend.position="none")

plot_grid(ggplot(all_data, aes(x=BusinessTravel,fill=Attrition))+ geom_bar(position = "fill"), 
          ggplot(all_data, aes(x=Department,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=EducationField,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=Gender,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=Education,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=JobRole,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          align = "h")   
#People in HR department and those with HR education field leave more often
#Also people who travel frequently are likelier to leave
#25% of Research Directors have left in the past year

plot_grid(ggplot(all_data, aes(x=factor(JobLevel),fill=Attrition))+ geom_bar(position = "fill"), 
          ggplot(all_data, aes(x=MaritalStatus,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=factor(NumCompaniesWorked),fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=factor(StockOptionLevel),fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          align = "h")   
#Single people are much more likely to leave
#Those who have already worked in >=5 companies are much more likely to leave

plot_grid(ggplot(all_data, aes(x=EnvironmentSatisfaction,fill=Attrition))+ geom_bar(position = "fill"), 
          ggplot(all_data, aes(x=JobSatisfaction,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=WorkLifeBalance,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=JobInvolvement,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          ggplot(all_data, aes(x=PerformanceRating,fill=Attrition))+ geom_bar(position = "fill")+bar_theme1,
          align = "h")   
#When employees give more 1's in the Employee Survey, they are much more likely to leave

# Boxplots of numeric variables relative to Attrition status
plot_grid(ggplot(all_data, aes(x=Attrition,y=Age, fill=Attrition))+ geom_boxplot(width=0.2)+ 
            coord_flip() +theme(legend.position="none"),
          ggplot(all_data, aes(x=Attrition,y=DistanceFromHome, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          ggplot(all_data, aes(x=Attrition,y=MonthlyIncome, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          align = "v",nrow = 1)
# Younger people are more likely to leave 

plot_grid(ggplot(all_data, aes(x=Attrition,y=PercentSalaryHike, fill=Attrition))+ geom_boxplot(width=0.2)+ 
            coord_flip() +theme(legend.position="none"),
          ggplot(all_data, aes(x=Attrition,y=TotalWorkingYears, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          ggplot(all_data, aes(x=Attrition,y=TrainingTimesLastYear, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          align = "v",nrow = 1)
#People with less total experience are more likely to leave

plot_grid(ggplot(all_data, aes(x=Attrition,y=YearsAtCompany, fill=Attrition))+ geom_boxplot(width=0.2)+ 
            coord_flip() +theme(legend.position="none"),
          ggplot(all_data, aes(x=Attrition,y=YearsSinceLastPromotion, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          ggplot(all_data, aes(x=Attrition,y=YearsWithCurrManager, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          ggplot(all_data, aes(x=Attrition,y=AvgTime, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          ggplot(all_data, aes(x=Attrition,y=leaveCount, fill=Attrition))+ geom_boxplot(width=0.2)+
            coord_flip() + box_theme_y,
          align = "v",nrow = 1)
#People with less years at XYZ company and less years with current manager are more likely to leave
#People who spend more time at office are likelier to leave

# Correlation between numeric variables
ggpairs(all_data[, c("Age", "DistanceFromHome", "MonthlyIncome", "PercentSalaryHike", 
                     "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
                     "YearsSinceLastPromotion", "YearsWithCurrManager", "AvgTime", "leaveCount")])
#No unexpected correlations

#--------------- Feature standardisation And Data Conversion -----------------

# converting target variable Attrition from No/Yes character levels 0/1 
all_data$Attrition<- ifelse(all_data$Attrition=="Yes",1,0)

# Checking Attrition rate 
Attrition <- sum(all_data$Attrition)/nrow(all_data)
Attrition # 16.12% Attrition rate. 

#Numeric columns (scaling):
#Age, DistanceFromHome, MonthlyIncome, NumCompaniesWorked, 
#PercentSalaryHike, TotalWorkingYears, TrainingTimesLastYear,
#YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, AvgTime, leaveCount

all_data$Age<- scale(all_data$Age) 
all_data$DistanceFromHome<- scale(all_data$DistanceFromHome) 
all_data$MonthlyIncome<- scale(all_data$MonthlyIncome) 
all_data$PercentSalaryHike<- scale(all_data$PercentSalaryHike) 
all_data$TotalWorkingYears<- scale(all_data$TotalWorkingYears) 
all_data$TrainingTimesLastYear<- scale(all_data$TrainingTimesLastYear) 
all_data$YearsAtCompany<- scale(all_data$YearsAtCompany) 
all_data$YearsSinceLastPromotion<- scale(all_data$YearsSinceLastPromotion) 
all_data$YearsWithCurrManager<- scale(all_data$YearsWithCurrManager) 
all_data$AvgTime <- scale(all_data$AvgTime)
all_data$leaveCount <- scale(all_data$leaveCount)

#Categorical but stored as numeric (We need to convert them to categorical first)
#Education (Done at time of EDA), JobLevel, StockOptionLevel, EnvironmentSatisfaction, 
#JobSatisfaction, WorkLifeBalance, JobInvolvement, PerformanceRating

all_data$JobLevel <- as.character(all_data$JobLevel)
all_data$StockOptionLevel <- as.character(all_data$StockOptionLevel)

all_data$EnvironmentSatisfaction <- ifelse(all_data$EnvironmentSatisfaction==1, "Low",
                                           ifelse(all_data$EnvironmentSatisfaction==2, "Medium",
                                                  ifelse(all_data$EnvironmentSatisfaction==3, "High", "Very High")))

all_data$JobSatisfaction <- ifelse(all_data$JobSatisfaction==1, "Low",
                                   ifelse(all_data$JobSatisfaction==2, "Medium",
                                          ifelse(all_data$JobSatisfaction==3, "High", "Very High")))

all_data$WorkLifeBalance <- ifelse(all_data$WorkLifeBalance==1, "Bad",
                                   ifelse(all_data$WorkLifeBalance==2, "Good",
                                          ifelse(all_data$WorkLifeBalance==3, "Better", "Best")))

all_data$JobInvolvement <- ifelse(all_data$JobInvolvement==1, "Low",
                                  ifelse(all_data$JobInvolvement==2, "Medium",
                                         ifelse(all_data$JobInvolvement==3, "High", "Very High")))

all_data$PerformanceRating <- ifelse(all_data$PerformanceRating==1, "Low",
                                     ifelse(all_data$PerformanceRating==2, "Good",
                                            ifelse(all_data$PerformanceRating==3, "Excellent", "Outstanding")))

#Rest of the Categorical columns:
#Attrition, BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus

# creating a dataframe of categorical features
categorical_data <- all_data[,-c(1,2,5,12,13,14,16,17,18,19,20,26,27)]

# converting categorical attributes to factor
all_data_fact<- data.frame(sapply(categorical_data, function(x) factor(x)))
str(all_data_fact)

# creating dummy variables for factor attributes
dummies <- data.frame(sapply(all_data_fact, 
                             function(x) data.frame(model.matrix(~x-1, data = all_data_fact))[,-1]))

# Final dataset
all_data_final<- cbind(all_data[,c(1,2,5,12,13,14,16,17,18,19,20,26,27)],dummies) 

str(all_data_final) #57 variables

#--------------------------- Model Building ---------------------------

#splitting the data between train and test
set.seed(100)

# randomly generate row indices for train dataset
trainindices= sample(1:nrow(all_data_final), 0.7*nrow(all_data_final))

# generate the train data set
train = all_data_final[trainindices,]
#Similarly store the rest of the observations into an object "test".
test = all_data_final[-trainindices,]

#Logistic Regression: 

#Initial model
model_1 = glm(Attrition ~ ., data = train, family = "binomial")
summary(model_1) #AIC 2157.9 Null deviance: 2747.7 Residual deviance: 2043.9

# Stepwise selection
model_2<- stepAIC(model_1, direction="both")
summary(model_2) #AIC 2127.3

#Model given by stepAIC
model_3<- glm(formula = Attrition ~ Age + MonthlyIncome + NumCompaniesWorked + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                Department.xSales + Education.xDoctor + EducationField.xLife.Sciences + 
                EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                EducationField.xTechnical.Degree + JobLevel.x5 + JobRole.xLaboratory.Technician + 
                JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                JobRole.xResearch.Scientist + JobRole.xSales.Executive + 
                MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                WorkLifeBalance.xGood + JobInvolvement.xLow + JobInvolvement.xMedium + 
                JobInvolvement.xVery.High, family = "binomial", data = train) 
summary(model_3) 

#JobInvolvementVery.High has a very low p-value (>0.1) so it can be removed
model_4<- glm(formula = Attrition ~ Age + MonthlyIncome + NumCompaniesWorked + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                Department.xSales + Education.xDoctor + EducationField.xLife.Sciences + 
                EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                EducationField.xTechnical.Degree + JobLevel.x5 + JobRole.xLaboratory.Technician + 
                JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                JobRole.xResearch.Scientist + JobRole.xSales.Executive + 
                MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                WorkLifeBalance.xGood + JobInvolvement.xLow + JobInvolvement.xMedium, 
              family = "binomial", data = train) 
summary(model_4)

#JobInvolvementMedium has a very low p-value (>0.09) so it can be removed
model_5 <- glm(formula = Attrition ~ Age + MonthlyIncome + NumCompaniesWorked + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                 YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xDoctor + EducationField.xLife.Sciences + 
                 EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                 EducationField.xTechnical.Degree + JobLevel.x5 + JobRole.xLaboratory.Technician + 
                 JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                 JobRole.xResearch.Scientist + JobRole.xSales.Executive + 
                 MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                 EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                 JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                 WorkLifeBalance.xGood + JobInvolvement.xLow, 
               family = "binomial", data = train) 
summary(model_5)

#JobInvolvementLow has a very low p-value (>0.1) so it can be removed
model_6 <- glm(formula = Attrition ~ Age + MonthlyIncome + NumCompaniesWorked + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                 YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xDoctor + EducationField.xLife.Sciences + 
                 EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                 EducationField.xTechnical.Degree + JobLevel.x5 + JobRole.xLaboratory.Technician + 
                 JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                 JobRole.xResearch.Scientist + JobRole.xSales.Executive + 
                 MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                 EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                 JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                 WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_6)

#JobRoleLaboratory.Technician has a very low p-value (>0.1) so it can be removed
model_7 <- glm(formula = Attrition ~ Age + MonthlyIncome + NumCompaniesWorked + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                 YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xDoctor + EducationField.xLife.Sciences + 
                 EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                 EducationField.xTechnical.Degree + JobLevel.x5 +  
                 JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                 JobRole.xResearch.Scientist + JobRole.xSales.Executive + 
                 MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                 EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                 JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                 WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_7)

#JobRoleResearch.Scientist has a very low p-value (>0.1) so it can be removed
model_8 <- glm(formula = Attrition ~ Age + MonthlyIncome + NumCompaniesWorked + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                 YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xDoctor + EducationField.xLife.Sciences + 
                 EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                 EducationField.xTechnical.Degree + JobLevel.x5 +  
                 JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                 JobRole.xSales.Executive + 
                 MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                 EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                 JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                 WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_8)

#MonthlyIncome has a very low p-value (>0.06) so it can be removed
model_9 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                 TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                 YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                 BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                 Department.xSales + Education.xDoctor + EducationField.xLife.Sciences + 
                 EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                 EducationField.xTechnical.Degree + JobLevel.x5 +  
                 JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                 JobRole.xSales.Executive + 
                 MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                 EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                 JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                 WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_9)

#EducationFieldLife.Sciences has a very low p-value (>0.06) so it can be removed
model_10 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + 
                  EducationField.xMarketing + EducationField.xMedical + EducationField.xOther + 
                  EducationField.xTechnical.Degree + JobLevel.x5 +  
                  JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                  JobRole.xSales.Executive + 
                  MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_10)

#EducationFieldMarketing has a very low p-value (>0.4) so it can be removed
model_11 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + 
                  EducationField.xMedical + EducationField.xOther + 
                  EducationField.xTechnical.Degree + JobLevel.x5 +  
                  JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                  JobRole.xSales.Executive + 
                  MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_11)

#EducationFieldOther has a very low p-value (>0.3) so it can be removed
model_12 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + 
                  EducationField.xMedical + 
                  EducationField.xTechnical.Degree + JobLevel.x5 +  
                  JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                  JobRole.xSales.Executive + 
                  MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_12)

#EducationFieldMedical has a very low p-value (>0.2) so it can be removed
model_13 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + 
                  EducationField.xTechnical.Degree + JobLevel.x5 +  
                  JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                  JobRole.xSales.Executive + 
                  MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_13)

#EducationFieldTechnical.Degree has a very low p-value (>0.1) so it can be removed
model_14 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + JobLevel.x5 +  
                  JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                  JobRole.xSales.Executive + 
                  MaritalStatus.xSingle + StockOptionLevel.x1 + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_14)

#StockOptionLevel1 has a very low p-value (>0.06) so it can be removed
model_15 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + JobLevel.x5 +  
                  JobRole.xManufacturing.Director + JobRole.xResearch.Director + 
                  JobRole.xSales.Executive + 
                  MaritalStatus.xSingle + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_15)

#JobRoleResearch.Director has a very low p-value (>0.05) so it can be removed
model_16 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + JobLevel.x5 +  
                  JobRole.xManufacturing.Director + JobRole.xSales.Executive + 
                  MaritalStatus.xSingle + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_16)

#JobRoleSales.Executive has a very low p-value (>0.06) so it can be removed
model_17 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + JobLevel.x5 +  
                  JobRole.xManufacturing.Director +  
                  MaritalStatus.xSingle + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_17)

# Removing multicollinearity through VIF check
vif(model_17)

# Variables with high multi-collinearity are also highly significant
#So let's remove the variable with the highest p-value first.
#JobLevel5 with p-value > 0.049
model_18 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + Education.xDoctor + JobRole.xManufacturing.Director +  
                  MaritalStatus.xSingle + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 

summary(model_18)
vif(model_18)

# Variables with high multi-collinearity are also highly significant
# So let's remove the variable with the highest p-value first.
# EducationDoctor with p-value > 0.04
model_19 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  BusinessTravel.xTravel_Rarely + Department.xResearch...Development + 
                  Department.xSales + JobRole.xManufacturing.Director +  
                  MaritalStatus.xSingle + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 

summary(model_19) #2139.4
vif(model_19)

#Now all variables are highly significant so let's try to remove one with high VIF and high p-value
#BusinessTravelTravel_Rarely has VIF  > 4.9 and p-value 0.000287
model_20 <- glm(formula = Attrition ~ Age + NumCompaniesWorked + 
                  TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                  YearsWithCurrManager + AvgTime + BusinessTravel.xTravel_Frequently + 
                  Department.xResearch...Development + 
                  Department.xSales + JobRole.xManufacturing.Director +  
                  MaritalStatus.xSingle + EnvironmentSatisfaction.xLow + 
                  EnvironmentSatisfaction.xVery.High + JobSatisfaction.xLow + 
                  JobSatisfaction.xVery.High + WorkLifeBalance.xBest + WorkLifeBalance.xBetter + 
                  WorkLifeBalance.xGood, family = "binomial", data = train) 
summary(model_20)
vif(model_20) 
#AIC 2153.3 #AIC has increased by 13.9
#So let's go back to previous model (model_19) which has AIC value of 2139.4 and
#VIF values are not too high for any variable

#With 20 significant variables in the model
summary(model_19)
final_model<- model_19

#--------------------------- Model Evaluation ---------------------------

#predicted probabilities of Attrition for test data
test_pred = predict(final_model, type = "response", 
                    newdata = test[,-2])
test$prob <- test_pred

# Let's use the probability cutoff of 50%.
test_pred_attrition <- factor(ifelse(test_pred >= 0.50, "Yes", "No"))
test_actual_attrition <- factor(ifelse(test$Attrition==1,"Yes","No"))

table(test_actual_attrition,test_pred_attrition)

#85.7 % Accuracy
#30 % Sensitivity
#96 % Specificity

#Sensitivity is very low for 50% cutoff so let's find an optimal cutoff value

#This function returns the values of accuracy, specificity and sensitivity for various cutoffs
perform_fn <- function(cutoff) 
{
  predicted_attrition <- factor(ifelse(test_pred >= cutoff, "Yes", "No"))
  conf <- confusionMatrix(predicted_attrition, test_actual_attrition, positive = "Yes")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  out <- t(as.matrix(c(sens, spec, acc))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}

# Summary of test probability
summary(test_pred)

#Vector containing various cutoff values
s = seq(.01,.80,length=100)
OUT = matrix(0,100,3)

#Let's see the values of accuracy, sensitivity and specificity for a range of cutoffs
for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
} 

#Plotting all the specificity, sensitivity and accuracy values
plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))

#Find the optimal cutoff value
cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.01)] # ~0.1935

# Let's choose a cutoff value of 0.1935 for final model
test_cutoff_attrition <- factor(ifelse(test_pred >=0.1935, "Yes", "No"))
conf_final <- confusionMatrix(test_cutoff_attrition, test_actual_attrition, positive = "Yes")

acc <- conf_final$overall[1] #75.1 % Accuracy
sens <- conf_final$byClass[1] #75.4 % Sensitivity
spec <- conf_final$byClass[2] #75.1 % Specificity

#-------------------------- KS-statistic --------------------------

#Convert the attrition values back to 1 and 0 for calculation
test_cutoff_attrition <- ifelse(test_cutoff_attrition=="Yes",1,0)
test_actual_attrition <- ifelse(test_actual_attrition=="Yes",1,0)

#Find the KS-statistic using R's inbuilt functions 
pred_object_test<- prediction(test_cutoff_attrition, test_actual_attrition)
performance_measures_test<- performance(pred_object_test, "tpr", "fpr")
ks_table_test <- attr(performance_measures_test, "y.values")[[1]] - 
  (attr(performance_measures_test, "x.values")[[1]])

max(ks_table_test) #KS-Statistics = 50.45 %

#-------------------------- Lift & Gain Chart --------------------------

#Function to calculate Gain and Lift
lift <- function(labels , predicted_prob,groups=10) {
  
  if(is.factor(labels)) labels  <- as.integer(as.character(labels ))
  if(is.factor(predicted_prob)) predicted_prob <- as.integer(as.character(predicted_prob))
  helper = data.frame(cbind(labels , predicted_prob))
  helper[,"bucket"] = ntile(-helper[,"predicted_prob"], groups)
  gaintable = helper %>% group_by(bucket)  %>%
    summarise_at(vars(labels ), funs(total = n(),
                                     totalresp=sum(., na.rm = TRUE))) %>%
    
    mutate(Cumresp = cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups))) 
  return(gaintable)
}

attrition_decile = lift(test_actual_attrition, test_pred, groups = 10)
#Add random model's gain % for comparitive plotting
attrition_decile$randomGain <- seq(10,100,10)

#Plot Gain Chart - our model's and random model's Gain % for each decile
ggplot(attrition_decile, aes(x = factor(bucket), y=Gain, group=1, label=round(Gain,2))) + 
  geom_point(colour="blue") + geom_line(colour="blue") + 
  geom_point(aes(y = randomGain), colour = "red") +
  geom_line(aes(y = randomGain), colour = "red") +
  labs(x="Decile", y="Gain % for our Model (Blue) & Random Model (Red)") +
  geom_text(aes(label=round(Gain,2)),hjust=0, vjust=1)

# plotting the lift chart
ggplot(attrition_decile, aes(x = factor(bucket), y=Cumlift, group=1, label=round(Cumlift,2))) + 
  geom_point(colour="blue") + geom_line(colour="blue") +
  labs(x="Decile", y="Lift") +
  geom_text(aes(label=round(Cumlift,2)),hjust=0, vjust=-1)

#-------------------------- Conclusions --------------------------------
#Our final model (with a cutoff of 0.1935) gives the following evaluation metrics:
#75.1 % Accuracy
#75.4 % Sensitivity
#75.1 % Specificity
#50.45 KS-Statistic
#Good Gains and Lifts

#The following factors were identified having an impact on attrition (with their coefficients):
#Age                                -0.33873
#NumCompaniesWorked                  0.16077    
#TotalWorkingYears                  -0.48231    
#TrainingTimesLastYear              -0.20746    
#YearsSinceLastPromotion             0.53041    
#YearsWithCurrManager               -0.49277    
#AvgTime                             0.68415    
#BusinessTravel.xTravel_Frequently   1.77261    
#BusinessTravel.xTravel_Rarely       0.97904    
#Department.xResearch...Development -1.13425    
#Department.xSales                  -1.20329    
#JobRole.xManufacturing.Director    -0.94010    
#MaritalStatus.xSingle               0.93899    
#EnvironmentSatisfaction.xLow        0.78452    
#EnvironmentSatisfaction.xVery.High -0.47959    
#JobSatisfaction.xLow                0.46658    
#JobSatisfaction.xVery.High         -0.75023    
#WorkLifeBalance.xBest              -1.07653    
#WorkLifeBalance.xBetter            -1.22093    
#WorkLifeBalance.xGood              -1.00211

#To sum it up:
#Employees working in HR are likelier to leave
#Employees who gave low ratings in surveys w.r.t work-life balance, job satisfaction or work environment satisfaction are likelier to leave
#Employees who travel are likelier to leave
#Young, single and less experienced employees are likelier to leave
#Employees who spend more time in the office are likelier to leave
#People who have not been promoted in many years are likelier to leave
#If employees have been with the same manager for many years, they are less likely to leave
#People who have already worked for a number of companies are likelier to leave
#Employees who attended fewer trainings are likelier to leave
#Employees in the role of Manufacturing Director are less likely to leave