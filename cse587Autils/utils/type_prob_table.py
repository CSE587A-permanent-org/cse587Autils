import pandas as pd

def typeProbTable(binCountsList, posteriors):
    trials = len(binCountsList[0])
    data = binCountsList + [posteriors] + [[1 - p for p in posteriors]]

    df = pd.DataFrame(data).transpose()
    df.columns = ['Trial {}'.format(i+1) for i in range(trials)] + ['Pr(dieType=1|trial)', 'Pr(dieType=2|trial)']
    
    df.loc['Total', :] = df.sum(axis=0)
    
    return df
