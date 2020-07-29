import pandas as pd
from efficient_apriori import apriori

# 数据加载
data = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# print(data.shape) # (7501, 20)
# transactions
transactions = []
for i in range(0,data.shape[0]):
    temp = []
    for j in range(0,data.shape[1]):
        if str(data.values[i, j])!= 'nan':
            temp.append(str(data.values[i, j]))
    transactions.append(temp)
# print(transactions)
# 挖掘频繁项集和关联规则

itemsets, rules = apriori(transactions, min_support=0.02, min_confidence=0.4)
print('频繁项集:', itemsets)
print('关联规则:', rules)
