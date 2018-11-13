# This python file perform the rule based ensemble

import pandas as pd


def main():
    m1 = pd.read_csv('result-model-1.txt')
    m2 = pd.read_csv('result-model-2.txt')
    print('result model1 shape: {}, result model2 shape: {}'.format(m1.shape, m2.shape))
    m1_result, m2_result = m1['category'].values, m2['category'].values
    diverge_count = len([index for index, item in enumerate(m1_result) if m2_result[index] != item])
    print('There are {} divergence between model1 and model2, i.e. {}%'.format(diverge_count, float(diverge_count) / len(m1_result)))

    when_listen_to_m2 = [0, 1, 3]
    '''
    It means that when m1 predict to 4, and m2 predict to following label,
    it will listen to m2 as final prediction.
    You can tune this parameter
    when we submit to kaggle, we use
    when_listen_to_m2 = [0, 1, 3], it reaches as our best performance
    on private leaderboad at 0.89649

    we also try when_listen_to_m2 = [0, 1, 2, 3], with only 0.89336

    we guess the reason is due to the model can hardly distinguish 2 from 4
    even with the help of webpage content. so it is better to listen to primary model 1
    '''

    result = []
    for i, item in enumerate(m1_result):
        if m1_result[i] == 4 and m2_result[i] in when_listen_to_m2:
            result.append(m2_result[i])
            continue
        result.append(m1_result[i])  # Yes, our primary model is m1, but you can also try m2.
    print(len(result))

    file_name = 'result-ensemble.txt'
    write_file = open(file_name, 'w+')
    write_file.write('article_id,category\n')
    i = 1
    for item in result:
        write_file.write('{},{}\n'.format(i, item))
        i += 1
    write_file.close()
    print('Have saved result to {}.'.format(file_name))


main()
