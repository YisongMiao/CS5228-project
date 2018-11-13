'''
In this python file, you can see that,
we successfully replicate our experiment result submitted to kaggle.
'''


def main():
    now = open(('result-ensemble.txt'), 'r')
    kaggle_best = open(('kaggle-best.txt'), 'r')
    r1 = now.readlines()
    r2 = kaggle_best.readlines()

    diverge = 0
    for i, item in enumerate(r1):
        if item != r2[i]:
            diverge += 1
    print('The divergence between today result and kaggle-best is: {} instances.'.format(diverge))


main()
