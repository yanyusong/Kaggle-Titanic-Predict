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

train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/train.csv")
test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/test.csv")
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
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/test.csv")
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
    df.loc[df['Age']<10,'Age']=0
    df.loc[(df['Age']>=10) & (df['Age']<20),'Age']=1
    df.loc[(df['Age']>=20) & (df['Age']<30),'Age']=2
    df.loc[(df['Age']>=30) & (df['Age']<40),'Age']=3
    df.loc[(df['Age']>=40) & (df['Age']<50),'Age']=4
    df.loc[(df['Age']>=50) & (df['Age']<60),'Age']=5
    df.loc[(df['Age']>=60) & (df['Age']<70),'Age']=6
    df.loc[(df['Age']>=70) & (df['Age']<80),'Age']=7
    df.loc[df['Age']>=80,'Age']=8

    temp = 0
    if title !='Rare':
        temp = df[(df['Age'].notnull())&(df['Title']==title)]['Age'].mode()
    else:
        temp = df[(df['Age'].notnull())&(df['Sex']==sex)]['Age'].mode()
    # print("---Age---%d"%temp[0])
    return temp[0]

# 获取fare band num 的众数
def get_farebandnum():
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/test.csv")
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
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/test.csv")
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
    df.loc[df['Age']<10,'Age']=0
    df.loc[(df['Age']>=10) & (df['Age']<20),'Age']=1
    df.loc[(df['Age']>=20) & (df['Age']<30),'Age']=2
    df.loc[(df['Age']>=30) & (df['Age']<40),'Age']=3
    df.loc[(df['Age']>=40) & (df['Age']<50),'Age']=4
    df.loc[(df['Age']>=50) & (df['Age']<60),'Age']=5
    df.loc[(df['Age']>=60) & (df['Age']<70),'Age']=6
    df.loc[(df['Age']>=70) & (df['Age']<80),'Age']=7
    df.loc[df['Age']>=80,'Age']=8
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
    df['Sex*Pclass'] = df['Sex'] * df['Pclass']
    # df['Sex*Embarked'] = df['Sex'] * df['Embarked']
    # df['Embarked*Pclass'] = df['Embarked'] * df['Pclass']

    return df

# debug
# train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/train.csv")
# test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/test.csv")

# #drop
# #handle

# train_df = drop(train_df)
# test_df = drop(test_df)

# df = handleDf(test_df)
# df.info()
# df.head()

#%%
def main():
    train_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/train.csv")
    test_df = pd.read_csv("/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/test.csv")
    originCombine = [train_df,test_df] # list 类型，DataFrame

    combine = []
    for df in originCombine:
        df = handleDf(drop(df))
        combine.append(df)

    X_train = combine[0].drop('Survived',axis=1)
    Y_train = combine[0]['Survived']
    X_test = combine[1].copy()

    print(X_train.shape,Y_train.shape,X_test.shape)
    
    logreg = LogisticRegression()
    logreg.fit(X_train,Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train,Y_train)*100,8)
    print(acc_log)

    coeff_df = pd.DataFrame(X_train.columns)
    coeff_df.columns = ['Feature']
    coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
    # print(coeff_df.sort_values(by='Correlation',ascending=False))
    
    svc = SVC()
    svc.fit(X_train,Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train,Y_train)*100,8)
    print(acc_svc)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 8)
    print(acc_knn)

    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 8)
    print(acc_gaussian)

    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 8)
    print(acc_perceptron)

    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 8)
    print(acc_linear_svc)

    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 8)
    print(acc_sgd)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 8)
    print(acc_decision_tree)

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 8)
    print(acc_random_forest)

    models = pd.DataFrame({
        'Model':['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
        'Score':[acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]
    })
    print(models.sort_values(by='Score', ascending=False))
    
    submission = pd.DataFrame({
        'PassengerId':test_df['PassengerId'],
        'Survived':Y_pred # 取的 Random Forest 的 Y_pred
    })
    # submission.to_csv('/Users/mac/2017/MyBag17/MyProject/python/IPythonNotebooks/pandas_lesson2/submission1.csv',index=False)
#%%
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

# 下个版本 version
# 交叉验证集 
# 增加多项式防止过拟合
# 调参
# 评价标准，准确率，F score