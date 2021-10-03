def compute_avgs(dataframe, column, params, avg_column):
    dic = {}
    for param in params:
        avg = dataframe.loc[dataframe[column] == param, avg_column].mean()
        dic[param] = avg
    for key, item in dic.items():
        print(f'{key}: {item}')

def compute_best_values(dataframe):
    print(f'Fitness Avg: {dataframe.Fitness.mean()}')
    print(f'Fitness Max: {dataframe.Fitness.max()}')
    print(f'Time Avg: {dataframe.Time.mean()}')