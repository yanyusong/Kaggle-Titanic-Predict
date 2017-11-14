#%%
# 根据 https://www.kaggle.com/startupsci/titanic-data-science-solutions
# Version 15 来学习的
import pandas as pd
import numpy as np
import random
import functools

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron,SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing
import xgboost as xgb
# from sklearn.model_selection import train_test_split

train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/train.csv")
test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/test.csv")
#%%
'''
数据分析
数据预处理
结论：
所有特征：
['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']
确定有用：Pclass、Sex、Age、Fare
处理后可能有点用：Name、SibSp、Parch、Embarked
无用舍弃：PassengerId、Survived、Ticket、Cabin
'''
# 查看数据的一些基本信息，种类、size
train_df.columns.values
train_df.head()
train_df.info() 
test_df.info()
train_df.describe()
train_df.describe(include=['O'])

#%%
# 有关,没缺失值
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values('Survived',ascending=False)
#%%
# 有关
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values('Survived',ascending=False)
#%%
# g = sns.FacetGrid(train_df,col='Sex')
# g.map(plt.hist,'Survived',bins=10)

# g = sns.FacetGrid(train_df,col='Survived' ,hue='Sex')
# g = g.map(plt.scatter,'Age','Fare',edgecolor='w').add_legend()

#%%
# Sex 性别与存活死亡的关系
# 由图：可确认女性比男性有更大概率幸存,女性生存概率超70%，男性只有不到20%
sns.barplot('Sex','Survived',data=train_df[['Sex','Survived']])

# s = (train_df['Survived'][train_df['Sex'] == 'male']).dropna()
# ns = (train_df['Survived'][train_df['Sex'] == 'female']).dropna()
# plt.hist([s,ns],bins=3,stacked = True,label=['Sex=Male','Sex=Female'])
# plt.legend()
# plt.show()

#%%
# Age 各年龄段存活死亡的人数分布
# 设想：小孩、孕妇、老人优先，所以生存概率更大
# 由图：事实是： 
# 0-15左右的小孩确实有更大的可能幸存
# 但是50-65以后的老人并没有特别明显的大概率幸存
# 而65岁以后的老人几乎很小的概率幸存
# 结论：Age 与 Survived有相关性
s = (train_df['Age'][train_df['Survived'] == 1]).dropna()
ns = (train_df['Age'][train_df['Survived'] == 0]).dropna()
plt.hist([s,ns],bins=24,stacked = True,label=['Survived=1','Survived=0'])
plt.legend()
plt.xlabel("Age")
plt.ylabel("Num")
plt.show()
# 猜想：年龄要分区间计算
# 结论：由图年龄之间没有明显断层，几乎是连续的，分区间可按10年为间隔
# 【0，10，20，30，40，50，60，70，80，80以上】
df= train_df[['Age','Survived']]
sns.stripplot(x='Survived',y='Age',data=df)

#%%
# 一同上船的
# SibSp 平级亲人，兄弟姐妹或配偶 和 Parch 上下级亲人，父母或孩子
df = train_df[['SibSp','Parch','Survived']]
# df.info() # 发现没有空值
# df.describe()
sns.distplot(df[['SibSp']])
sns.distplot(df[['Parch']])

#%% 
#分析后可知：没有缺失值，891中唯一的有681种，感觉没啥相关性，舍弃
df = train_df[['Ticket','Survived']]
df.info()
df.describe(include=['O'])

#%%
# Fare
# 猜想：Fare 与 Survived是相关的
# 结论：由图可知：Fare 与 Survived有关，Fare 越高幸存概率越大
s = train_df['Fare'][train_df['Survived']==1]
ns = train_df['Fare'][train_df['Survived']==0]
sns.distplot(s,kde=False,bins=40,label='1')
sns.distplot(ns,kde=False,bins=40,label='0')
plt.legend()

#%%
# Fare
# 猜想：票价不同，可能会分几个档次，越贵船舱等级越高，即幸存几率越大
# 结论：由图大致分为：0-100，100-200，200-300，300以上
df= train_df[['Fare','Survived']]
sns.stripplot(x='Survived',y='Fare',data=df)

#%%
# Cabin
# 观察发现：共891，缺失值有891-204个，太多了，舍弃这个特征吧
df = train_df[['Cabin','Survived']]
df.head(40)
df.info()
# df.describe()
# 测试集中这个特征缺失值也很高，所以舍弃吧，它也没有什么有用的信息能挖掘
test_df.info()

#%%
# Embarked 上船的港口
# 猜想：Embarked 可能跟 Survived 有关，模型训练时增加删除它确认一下
# 哪个效果好
df = train_df[['Embarked','Survived']]
df.head(40)
df.info()

'''
['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']

通过以上分析我们决定对各个特征做以下特征工程处理：

无用舍弃：PassengerId、Survived、Ticket、Cabin
确定有用：Pclass、Sex、Age、Fare
处理后可能有点用：Name、SibSp、Parch、Embarked

舍弃：PassengerId、Survived、Ticket、Cabin
确定有用：
Pclass(不作处理)
Sex(二值化)
Age(1、补缺失值 2、band)
Fare(1、补缺失值 2、band)

处理后可能有点用：
Name(1.根据name补全age的缺失值 2.转换成新feature Title)
SibSp、Parch (合并成新的feature FamilySize)
Embarked(1.补缺失值 2、转换成数字序号)
'''
#%%
# 前边的分析阶段不要对数据集做任何改变
# 特征工程开始时把数据集合并,方便做统一的处理清洗转换操作
train_df.info()
test_df.info()
train_df.columns.values
print(len(train_df.columns.values))
print(len(test_df.columns.values))
#%% 
# 定义一个方法装饰器，方便将特征工程的处理同时统一应用到训练集和测试集合
originConbine = [train_df,test_df] # list 类型，DataFrame

def applyToConbine(dfList=[]):
    def decorater(func):
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            dfs = []
            for df in dfList:
                dfs.append(func(df))
            return dfs # 返回处理后的conbine数据
        return wrapper
    return decorater
#%%
# 舍弃掉无用的
# PassengerId、Survived、Ticket、Cabin
def drop(df):
    df = df.drop(['PassengerId','Ticket','Cabin'],axis=1) # 沿着列排列的方向水平执行方法
    # if not istraindf:
    #     if 'Survived' in df.columns.values:
    #         df = df.drop('Survived',axis=1)
    return df

#%%
# 根据title获取全部数据中age的众数
# sex 只有在 title == Rare 时才有效
# Age (1、补缺失值 2、band)
# 补缺失值,根据Name生成Title，Mr取Mr里的众数，Mrs取Mrs里的众数，
# Miss取Miss里的众数，Master取Master里的众数
# Rare里的Age，因为Rare本身就比较稀少，取众数容易有很大误差，
# 所以Rare里的male取所有male里的众数，female取所有female里的众数
def get_agebandnum_by_title(title,sex=None):
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/test.csv")
    df = train_df.append(test_df)

    df['Title'] = df['Name'].str.extract('([a-zA-Z]+)\.',expand=False)
    # print(pd.crosstab(df['Title'],df['Sex']))
    df['Title'].replace(['Capt','Col','Countess','Don',\
    'Dr','Jonkheer','Lady','Major','Rev','Sir','Dona'],'Rare',inplace=True)
    df['Title'].replace('Mlle','Miss',inplace=True)
    df['Title'].replace('Ms','Miss',inplace=True)
    df['Title'].replace('Mme','Mrs',inplace=True)

    df.drop('Name',axis=1,inplace=True)
    # 将 Age 分组用序号替换
    # df.loc[df['Age']<10,'Age']=0
    # df.loc[(df['Age']>=10) & (df['Age']<20),'Age']=1
    # df.loc[(df['Age']>=20) & (df['Age']<30),'Age']=2
    # df.loc[(df['Age']>=30) & (df['Age']<40),'Age']=3
    # df.loc[(df['Age']>=40) & (df['Age']<50),'Age']=4
    # df.loc[(df['Age']>=50) & (df['Age']<60),'Age']=5
    # df.loc[(df['Age']>=60) & (df['Age']<70),'Age']=6
    # df.loc[(df['Age']>=70) & (df['Age']<80),'Age']=7
    # df.loc[df['Age']>=80,'Age']=8

    df.loc[df['Age']<10,'Age']=0
    df.loc[(df['Age']>=10) & (df['Age']<20),'Age']=1
    df.loc[(df['Age']>=20) & (df['Age']<50),'Age']=2
    df.loc[(df['Age']>=50) & (df['Age']<70),'Age']=3
    df.loc[df['Age']>=70,'Age']=4

    temp = 0
    if title !='Rare':
        temp = df[(df['Age'].notnull())&(df['Title']==title)]['Age'].mode()
    else:
        temp = df[(df['Age'].notnull())&(df['Sex']==sex)]['Age'].mode()
    # print("---Age---%d"%temp[0])
    return temp[0]

# 获取fare band num 的众数
def get_farebandnum():
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/test.csv")
    df = train_df.append(test_df)

    df.loc[df['Fare']<=100,'Fare']=0
    df.loc[(df['Fare']>100) & (df['Fare']<=200),'Fare']=1
    df.loc[(df['Fare']>200) & (df['Fare']<=300),'Fare']=2
    df.loc[(df['Fare']>300),'Fare']=3
    temp = df[df['Fare'].notnull()]['Fare'].mode()
    # print("---Fare---%d"%temp[0])
    return temp[0]

# 获取embarked num 的众数
def get_embarkednum():
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/test.csv")
    df = train_df.append(test_df)

     # Embarked(1.补缺失值 2、转换成数字序号)
    df.replace({'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
    temp = df['Embarked'].dropna().mode()
    # print("---Embarked---%d"%temp[0])
    return temp[0]

#%% 
def handleDf(df):
    # Sex 二值化
    df['Sex'].replace('male',1,inplace=True)
    df['Sex'].replace('female',0,inplace=True)
    # print(df['Sex'].describe(include='O'))

    # Age (1、补缺失值 2、band)
    # 1.补缺失值,根据Name生成Title，Mr取Mr里的众数，Mrs取Mrs里的众数，
    # Miss取Miss里的众数，Master取Master里的众数
    # Rare里的male Age取所有male里的众数，female的Age取所有female里的众数
    df['Title'] = df['Name'].str.extract('([a-zA-Z]+)\.',expand=False)
    # print(pd.crosstab(df['Title'],df['Sex']))
    df['Title'].replace(['Capt','Col','Countess','Don',\
    'Dr','Jonkheer','Lady','Major','Rev','Sir','Dona'],'Rare',inplace=True)
    df['Title'].replace('Mlle','Miss',inplace=True)
    df['Title'].replace('Ms','Miss',inplace=True)
    df['Title'].replace('Mme','Mrs',inplace=True)

    # print(pd.crosstab(df['Title'],df['Sex']))

    df.drop('Name',axis=1,inplace=True)
    # 将 Age 分组用序号替换
    # df.loc[df['Age']<10,'Age']=0
    # df.loc[(df['Age']>=10) & (df['Age']<20),'Age']=1
    # df.loc[(df['Age']>=20) & (df['Age']<30),'Age']=2
    # df.loc[(df['Age']>=30) & (df['Age']<40),'Age']=3
    # df.loc[(df['Age']>=40) & (df['Age']<50),'Age']=4
    # df.loc[(df['Age']>=50) & (df['Age']<60),'Age']=5
    # df.loc[(df['Age']>=60) & (df['Age']<70),'Age']=6
    # df.loc[(df['Age']>=70) & (df['Age']<80),'Age']=7
    # df.loc[df['Age']>=80,'Age']=8

    df.loc[df['Age']<10,'Age']=0
    df.loc[(df['Age']>=10) & (df['Age']<20),'Age']=1
    df.loc[(df['Age']>=20) & (df['Age']<50),'Age']=2
    df.loc[(df['Age']>=50) & (df['Age']<70),'Age']=3
    df.loc[df['Age']>=70,'Age']=4

    # temp = df[df['Age'].isnull()] # Age有缺失值的df
    # print(df['Age'].value_counts(dropna=False))

    # 根据 Title 补全 Age 里的缺失值
    df.loc[(df['Age'].isnull())&(df['Title']=='Master'),'Age']=get_agebandnum_by_title('Master')
    df.loc[(df['Age'].isnull())&(df['Title']=='Miss'),'Age']=get_agebandnum_by_title('Miss')
    df.loc[(df['Age'].isnull())&(df['Title']=='Mr'),'Age']=get_agebandnum_by_title('Mr')
    df.loc[(df['Age'].isnull())&(df['Title']=='Mrs'),'Age']=get_agebandnum_by_title('Mrs')
    df.loc[(df['Age'].isnull())&(df['Title']=='Rare')&(df['Sex']==1),'Age']=get_agebandnum_by_title('Rare','male') 
    df.loc[(df['Age'].isnull())&(df['Title']=='Rare')&(df['Sex']==0),'Age']=get_agebandnum_by_title('Rare','female')
    df['Age'] = df['Age'].astype(int)
    # 将 title 转换为 序号
    df.replace({'Title':{'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Rare':4}},inplace=True)

    # Fare
    df.loc[df['Fare']<=100,'Fare']=0
    df.loc[(df['Fare']>100) & (df['Fare']<=200),'Fare']=1
    df.loc[(df['Fare']>200) & (df['Fare']<=300),'Fare']=2
    df.loc[(df['Fare']>300),'Fare']=3
    df['Fare'].fillna(get_farebandnum(),inplace = True)
    df['Fare'] = df['Fare'].astype(int)

    # SibSp、Parch (合并成新的feature FamilySize) 
    # FamilySize
    # 加 1 是因为包含他自己
    df['FamilySize'] = df['SibSp']+df['Parch']+1 
    df.drop(['SibSp','Parch'],axis=1,inplace=True)

    # Embarked(1.补缺失值 2、转换成数字序号)
    df.replace({'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
    df['Embarked'].fillna(get_embarkednum(),inplace = True)
    df['Embarked'] = df['Embarked'].astype(int)
    # df = df.drop(['Embarked'],axis=1)
    # print(df['Embarked'].value_counts(dropna=False))

    # 根据 逻辑回归的 baseline 和 相关性分析
    # df['Sex*Pclass'] = df['Sex'] * df['Pclass']
    # df['Sex*Age'] = df['Sex'] * df['Age']
    # df['Sex*Embarked'] = df['Sex'] * df['Embarked']
    # df['Embarked*Pclass'] = df['Embarked'] * df['Pclass']

    return df

# debug
# train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/train.csv")
# test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/test.csv")

# #drop
# #handle

# train_df = drop(train_df)
# test_df = drop(test_df)

# df = handleDf(test_df)
# df.info()
# df.head()

#%%
# 画学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.05, 1., 20)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def cul_score(estimator,X,y,cv=4):
    # 交叉验证得分
    scores = cross_validation.cross_val_score(estimator,X,y,cv=cv)
    print(scores)
    print('Accuracy: %0.5f (+/- %0.2f)' % (scores.mean(),scores.std()*2))
    #F1 score
    scores = cross_validation.cross_val_score(estimator,X,y,cv=cv,scoring = 'f1_weighted')
    print(scores)
    print('F1 score : %0.5f (+/- %0.2f)' % (scores.mean(),scores.std()*2))
    
def modelfit(alg, dtrain_x,dtrain_y, dtest,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain_x.values, label=dtrain_y.values)
        xgtest = xgb.DMatrix(dtest.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain_x, dtrain_y,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain_x)
    dtrain_predprob = alg.predict_proba(dtrain_x)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain_y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain_y, dtrain_predprob))
    
#     Predict on testing data:
    # dtest['predprob'] = alg.predict_proba(dtest)[:,1]
    # results = test_results.merge(dtest[['ID','predprob']], on='ID')
    # print 'AUC Score (Test): %f' % metrics.roc_auc_score(results['Disbursed'], results['predprob'])
                
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

#%%
def main():
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/test.csv")
    originCombine = [train_df,test_df] # list 类型，DataFrame

    combine = []
    for df in originCombine:
        df = handleDf(drop(df))
        combine.append(df)

    X = combine[0].drop('Survived',axis=1)
    Y = combine[0]['Survived']
    X_test_origin = combine[1].copy()

    X = pd.get_dummies(X,columns=['Pclass','Sex','Embarked','Title'])
    X_test_origin = pd.get_dummies(X_test_origin,columns=['Pclass','Sex','Embarked','Title'])
    print(X.head(5))

    poly = preprocessing.PolynomialFeatures(2,interaction_only=False)
    poly.fit(X.values)
    X_train = pd.DataFrame(poly.transform(X.values),columns=poly.get_feature_names(X.columns))
    X_train=X_train.astype(int)

    # one-hot
    # onehotenc = preprocessing.OneHotEncoder()
    # onehotenc.fit(X)
    # X_train = onehotenc.transform(X).toarray()
    # X_train=pd.DataFrame(X_train)

    # X_train = X
    Y_train = Y
    # X_train,X_val,Y_train,Y_val = cross_validation.train_test_split(X,Y,random_state=0)
    # X_val = cross_validation.train_test_split() 
    # Y_val
    X_test = pd.DataFrame(poly.transform(X_test_origin.values),columns=poly.get_feature_names(X_test_origin.columns))
    X_test=X_test.astype(int)
    # X_test = X_test_origin
    # X_test = onehotenc.transform(X_test_origin).toarray()
    # X_test=pd.DataFrame(X_test)
    
    print(type(X_train))
    print(type(X_test))
    print(X_train.shape,Y_train.shape)
    print(X_train.head(2))
    print()
    # print(X_test.head(5))
    # print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)

    logreg = LogisticRegression()
    cul_score(logreg,X_train.values,Y_train.values)
    # 画学习曲线
    plot_learning_curve(logreg,'LogisticRegression Learning Curve',X_train.values,Y_train.values)
    plt.show()

    logreg.fit(X_train,Y_train)
    Y_pred_log = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train,Y_train)*100,8)
    print(acc_log)

    # print(type(X_train))
    # print(X_train.head(10))
    # print(type(Y_train))
    # xgboost
    # model = xgb.XGBClassifier(
    #     learning_rate =0.0997,
    #     n_estimators=130,
    #     max_depth=3,
    #     min_child_weight=1,
    #     gamma=0,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     objective= 'binary:logistic',
    #     nthread=4,
    #     scale_pos_weight=1,
    #     seed=27)

    # 调参 n_estimators
    # modelfit(model,X_train,Y_train,X_test)
    # 调参 max_depth 、min_child_weight
    # param_test1 = {
    #     'max_depth':range(3,10,2),
    #     'min_child_weight':range(1,6,2)
    # }
    # gsearch1 = GridSearchCV(estimator=model,param_grid=param_test1,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
    # gsearch1.fit(X_train,Y_train)
    # print(gsearch1.grid_scores_)
    # print(gsearch1.best_score_)
    # print(gsearch1.best_params_)

    # param_test2 = {
    #     'max_depth':[1,2,3,4],
    #     # 'min_child_weight':[4,5,6]
    # }
    # gsearch2 = GridSearchCV(estimator=model,param_grid=param_test2,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
    # gsearch2.fit(X_train,Y_train)
    # print(gsearch2.grid_scores_)
    # print(gsearch2.best_score_)
    # print(gsearch2.best_params_)


    # model.fit(X_train, Y_train)
    # Y_pred_xgb = model.predict(X_test)

    # submission = pd.DataFrame({
    #     'PassengerId':test_df['PassengerId'],
    #     'Survived':Y_pred_xgb 
    # })
    # # print(submission.head(20))
    # submission.to_csv('/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/submission_xgb.csv',index=False)

    # coeff_df = pd.DataFrame(X_train.columns)
    # coeff_df.columns = ['Feature']
    # coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
    # print(coeff_df.sort_values(by='Correlation',ascending=False))
    
    svc = SVC()

    cul_score(svc,X_train,Y_train)
     # 画学习曲线
    plot_learning_curve(logreg,'SVC Learning Curve',X_train,Y_train)
    plt.show()

    svc.fit(X_train,Y_train)
    Y_pred_svc = svc.predict(X_test)
    acc_svc = round(svc.score(X_train,Y_train)*100,8)
    print(acc_svc)

    # knn = KNeighborsClassifier(n_neighbors = 3)
    # knn.fit(X_train, Y_train)
    # Y_pred_knn = knn.predict(X_test)
    # acc_knn = round(knn.score(X_train, Y_train) * 100, 8)
    # print(acc_knn)

    # # gaussian = GaussianNB()
    # # gaussian.fit(X_train, Y_train)
    # # Y_pred = gaussian.predict(X_test)
    # # acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 8)
    # # print(acc_gaussian)

    # perceptron = Perceptron()
    # perceptron.fit(X_train, Y_train)
    # Y_pred_perceptron = perceptron.predict(X_test)
    # acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 8)
    # print(acc_perceptron)

    # linear_svc = LinearSVC()
    # linear_svc.fit(X_train, Y_train)
    # Y_pred_linear_svc = linear_svc.predict(X_test)
    # acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 8)
    # print(acc_linear_svc)

    # sgd = SGDClassifier()
    # sgd.fit(X_train, Y_train)
    # Y_pred_sgd = sgd.predict(X_test)
    # acc_sgd = round(sgd.score(X_train, Y_train) * 100, 8)
    # print(acc_sgd)


    # decision_tree = DecisionTreeClassifier()
    # decision_tree.fit(X_train, Y_train)
    # Y_pred_decision_tree = decision_tree.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 8)
    # print(acc_decision_tree)

    random_forest = RandomForestClassifier()

    # n_estimators=50,criterion="gini",max_depth=4

    # param_test = {
    #     'n_estimators':range(10,200,20),
    #     'criterion':['entropy','gini'],
    #     'max_depth':range(1,5,1),
    #     # 'min_samples_split':[2,5,10],
    #     # 'min_weight_fraction_leaf':[0.0,0.1,0.2,0.3,0.4,0.5]
    #     # 'min_child_weight':[4,5,6]
    # }
    # gsearch = GridSearchCV(estimator=random_forest,param_grid=param_test,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
    # gsearch.fit(X_train,Y_train)
    # print(gsearch.grid_scores_)
    # print(gsearch.best_score_)
    # print(gsearch.best_params_)

    # 画学习曲线
    plot_learning_curve(random_forest,'RandomForestClassifier Learning Curve',X_train,Y_train)
    plt.show()
    cul_score(random_forest,X_train,Y_train)

    random_forest.fit(X_train, Y_train)
    Y_pred_random_forest = random_forest.predict(X_test)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 8)
    print(acc_random_forest)

    # models = pd.DataFrame({
    #     'Model':['Support Vector Machines', 'KNN', 'Logistic Regression', 
    #           'Random Forest', 'Perceptron', 
    #           'Stochastic Gradient Decent', 'Linear SVC', 
    #           'Decision Tree'],
    #     'Score':[acc_svc, acc_knn, acc_log, 
    #           acc_random_forest,  acc_perceptron, 
    #           acc_sgd, acc_linear_svc, acc_decision_tree]
    # })
    # print(models.sort_values(by='Score', ascending=False))
    
    submission = pd.DataFrame({
        'PassengerId':test_df['PassengerId'],
        'Survived':Y_pred_log # 取的 Logistic Regression 的 Y_pred
    })
    submission.to_csv('/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/submission_log.csv',index=False)
   
    submission = pd.DataFrame({
        'PassengerId':test_df['PassengerId'],
        'Survived':Y_pred_svc # 取的 svc 的 Y_pred
    })
    submission.to_csv('/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/submission_svc.csv',index=False)

    submission = pd.DataFrame({
        'PassengerId':test_df['PassengerId'],
        'Survived':Y_pred_random_forest # 取的 Random Forest 的 Y_pred
    })
    submission.to_csv('/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/submission_rf.csv',index=False)
   
    # submission = pd.DataFrame({
    #     'PassengerId':test_df['PassengerId'],
    #     'Survived':Y_pred_svc # 取的 SVC 的 Y_pred
    # })
    # submission.to_csv('/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/submission_svc.csv',index=False)

    # 模型融合竟然没有用

    # Y_pre_cross = 0.6*Y_pred_random_forest+0.2*Y_pred_svc+0.2*Y_pred_log
    # print(Y_pre_cross)
    # submission = pd.DataFrame({
    # 'PassengerId':test_df['PassengerId'],
    # 'Survived':Y_pre_cross 
    # })
    # submission.loc[submission['Survived']>0.6,'Survived']=1
    # submission.loc[submission['Survived']<=0.6,'Survived']=0
    # submission['Survived'] = submission['Survived'].astype(int)
    # submission.info() 
    # print(submission.head(10))
    # submission.to_csv('/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/GitTitanicPredict/submission_cro.csv',index=False)

if __name__ == '__main__':
    main()


# test1
# 80.1347 logistic regression 
#       Feature  Correlation
# 4    Embarked     0.252617
# 3        Fare     0.098462
# 5       Title    -0.107733
# 6  FamilySize    -0.140020
# 2         Age    -0.220645
# 0      Pclass    -0.970048
# 1         Sex    -2.476070

# test3 Embarked*Pclass
# 80.35914703 logistic regression 
#            Feature  Correlation
# 4         Embarked     0.373952
# 3             Fare     0.087499
# 7  Embarked*Pclass    -0.048280
# 5            Title    -0.111333
# 6       FamilySize    -0.141588
# 2              Age    -0.221229
# 0           Pclass    -0.954066
# 1              Sex    -2.479831

# test4 Sex*Embarked
# 80.13468013 logistic regression 
#         Feature  Correlation
# 4      Embarked     0.450625
# 3          Fare     0.109716
# 5         Title    -0.088333
# 6    FamilySize    -0.129747
# 2           Age    -0.222978
# 7  Sex*Embarked    -0.383995
# 0        Pclass    -0.976639
# 1           Sex    -2.334776

# test2 Sex*Pclass
# 80.35914703 logistic regression 
# 86.53198653 random forest 
#       Feature  Correlation
# 4    Embarked     0.266603
# 7  Sex*Pclass     0.236189
# 3        Fare     0.084895
# 5       Title    -0.111686
# 6  FamilySize    -0.129086
# 2         Age    -0.213930
# 0      Pclass    -1.118597
# 1         Sex    -2.991686


# 交叉验证集 
# 增加多项式
# 调参
# 评价标准，准确率，F score

# 根据学习曲线，发现训练数据增多时精度并没有明显变化，分析是欠拟合
# one-hot 编码，因为：类别并不是连续的，详细见sklearn onhotencoding 的文档

# OneHotEncoding 对非树形算法有很大提升，比如逻辑回归
# 但是却让 random forest 过拟合了，随机森林不适合onehot
# test2 Sex*Pclass
# logistic regression onehot 对提升很大
# 之前F1=0.78929 (+/- 0.05),
# 之后F1=0.81637 (+/- 0.05)
# random forest onehot 对树模型没有太大提升
# 之前 F1=0.78040 (+/- 0.09),
# 之后 F1=0.78643 (+/- 0.08)

# 11月14日更新
# age 年龄区间减少后 正则后逻辑回归的成绩提升到了 0.79425
# 下一步正则加上one-hot，应该会对逻辑回归有更大提升