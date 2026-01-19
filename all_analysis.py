import pandas as pd
import matplotlib.pyplot as plt

tt = pd.read_csv('Titanic.csv')
tt = pd.read_csv('Titanic.csv', index_col='PassengerId')

# фильтр по возрасту(схема)
tt = pd.read_csv('Titanic.csv')
tt = pd.read_csv('Titanic.csv', index_col='PassengerId')
plt.hist(tt['Age'].dropna(), bins=20, color='skyblue', edgecolor='black')

print(tt['Fare'].describe()) # фильтр по стоимости билета

f_p = tt[tt['Sex'] == 'female'] # фильтр по полу
print(f'женщин {len(f_p)}')
print(f_p.head())

y_s_p = tt[(tt['Sex'] == 'male') & (tt['Age'] < 32) & (tt['Survived'] == 1)] # фильтр по полу, возрасту и фактору выживания
print(f'Выжившие мужчины младше 32 лет: {len(y_s_p)}')
print(y_s_p.head())

s_f_c = tt[(tt['Survived'] == 1) & (tt['Pclass'].isin([1,2]))] # фильтр по фактору выживания и классу
print(f'Кол-во: {len(s_f_c)}')
print(s_f_c.head())

tt_emb = tt['Embarked'].unique() # уникальные значения
print(tt_emb)

surv = tt.groupby('Survived')[['Age', 'Fare', 'SibSp', 'Parch']].mean() # средние значения у выживших/невыживших
print(surv)

print(tt.groupby('Sex')['Age'].agg(['mean', 'median'])) # средние значения по соотношению пол-возраст

