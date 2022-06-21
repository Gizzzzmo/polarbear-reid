# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from reid.metric_logging import load_run_data
import glob

sns.set(style='ticks')
sns.set_style('darkgrid')

def perm_tuple(x, indices):
    return tuple(x[i] for i in indices)
# %%

dm, tm, ts, ms, pt, sh = load_run_data('logs/Run_7')
for dataset_name in dm:
    dm[dataset_name] = pd.read_csv(dm[dataset_name])
    dm[dataset_name].set_index('epoch', inplace=True)
    dm[dataset_name] = dm[dataset_name][~dm[dataset_name].index.duplicated(keep='first')]
    dm[dataset_name] = dm[dataset_name].loc[:, ~dm[dataset_name].columns.str.replace("(\.\d+)$", "").duplicated()]
# %%
ts
# %%
dm['6_vienna_pad']
# %%
sns.lineplot(x='epoch', y='kappa', data=dm['4_berlin_pad'], legend='full')
sns.lineplot(x='epoch', y='kappa', data=dm['6_vienna_pad'])
sns.lineplot(x='epoch', y='kappa', data=dm['8_mulhouse_pad'], legend='full')
# %%
sns.lineplot(data=dm['6_vienna_pad'].filter(regex=r'svm_accuracy_tracklet_2_').mean(axis=1))
sns.lineplot(data=dm['6_vienna_pad'].filter(regex=r'svm_accuracy_tracklet_4_').mean(axis=1))
sns.lineplot(data=dm['6_vienna_pad'].filter(regex=r'svm_accuracy_tracklet_8_').mean(axis=1))
plt.ylim((0, 1))
# %%
qg = dm['6_vienna_pad'].filter(regex=r'^([a-z]*_[a-z]*_._.)|epoch')
qg.columns = pd.MultiIndex.from_tuples([perm_tuple(c.split('_'), (1, 0, 2, 3)) for c in qg.columns])
# %%

qg['kappa']['logreg']['8'].loc[::5].T
# %%

sns.boxplot(data=qg['kappa']['logreg']['8'].loc[::5].T, showfliers=False)

# %%

toplot = pd.melt(qg['accuracy']['logreg']['8'].reset_index(), id_vars='epoch')
sns.lineplot(data=toplot, x='epoch', y='value')
# %%
toplot = pd.melt(qg['accuracy']['logreg']['4'].reset_index(), id_vars='epoch')
sns.lineplot(data=toplot, x='epoch', y='value')
toplot = pd.melt(qg['accuracy']['logreg']['2'].reset_index(), id_vars='epoch')
sns.lineplot(data=toplot, x='epoch', y='value')
plt.ylim((0, 1))
# %%
datasets = ['2_NBapril_pad', '4_berlin_pad', '6_vienna_pad', '8_mulhouse_pad']

blub = np.zeros((2, 5, len(datasets)), dtype=object)
blubtrain = np.zeros((2, 5, len(datasets)), dtype=object)

for run in glob.glob('logs/*'):
    dm, tm, ts, ms, pt, sh = load_run_data(run)
    if len(ts) != 1:
        continue
    pt = 1 if pt else 0
    dataset_name, fold = ts[0]
    print(dataset_name, fold, pt)
    data = pd.read_csv(dm[dataset_name])
    data.set_index('epoch', inplace=True)
    data = data[~data.index.duplicated(keep='first')]
    data = data.loc[:, ~data.columns.str.replace("(\.\d+)$", "").duplicated()]
    blub[pt, int(fold)-1, datasets.index(dataset_name)] = data
    
    tm.set_index('epoch', inplace=True)
    data = tm[~tm.index.duplicated(keep='first')]
    data = data.loc[:, ~data.columns.str.replace("(\.\d+)$", "").duplicated()]
    blubtrain[pt, int(fold)-1, datasets.index(dataset_name)] = data
# %%
combined_folds = pd.concat([blub[1, fold, 0].loc[0:51, ['accuracy']].rename({'accuracy': f'{fold}'}, axis=1) for fold in range(5)], axis=1)

combined_folds = combined_folds.loc[::5]
combined_folds.reset_index()

combined_folds
# %% 
pd.melt(combined_folds.reset_index(), id_vars='epoch')

sns.lineplot(data=pd.melt(combined_folds.reset_index(), id_vars='epoch'), x='epoch', y='value')

# %%

blub[1, 0, 0].loc[0:51, ['accuracy']].rename({'accuracy': f'accuracy_{0}'}, axis=1).loc[::5]
# %%
for i, dataset_name in enumerate(datasets):
    for j, pretrained in enumerate(['untrained', 'ImageNet']):
        for metric in ['accuracy', 'f1', 'kappa']:
            print(f'{dataset_name} {pretrained} {metric}:')
            comb_folds1 = pd.concat([blub[j, fold, i].loc[0:51, [metric]].rename({metric: f'{fold}'}, axis=1) for fold in range(5)], axis=1)
            
            melted1 = pd.melt(comb_folds1.reset_index(), id_vars='epoch')
            sns.lineplot(data=melted1, x='epoch', y='value', label=f'validation {metric}', ci='sd')
            
            if metric == 'accuracy':
                comb_folds2 = pd.concat(
                    [
                        blubtrain[j, fold, i].loc[0:51, ['accuracy']].rename({'accuracy': f'{fold}'}, axis=1)
                            for fold in range(5)
                    ], axis=1
                )
                
                melted2 = pd.melt(comb_folds2.reset_index(), id_vars='epoch')
                
                sns.lineplot(data=melted2, x='epoch', y='value', label=f'training accuracy', ci='sd')
            
            lower_limit = 0.3 if metric != 'kappa' else -0.2
            plt.ylim((lower_limit, 1.15))
            plt.savefig(f'plots/{metric}__{dataset_name}__{pretrained}__lineplot.png', dpi=300)
            plt.show()
            
            sns.boxplot(data=comb_folds1.loc[::5].T, showfliers=False, color='#77aaff')
            plt.ylim((lower_limit, 1.15))
            plt.savefig(f'plots/{metric}__{dataset_name}__{pretrained}__boxplot.png', dpi=300)
            plt.show()
            
            
            sns.lineplot(data=melted1, x='epoch', y='value', label=f'validation {metric}', ci='sd')
            comb_folds3 = pd.concat([blub[j, fold, i].loc[0:51, [metric+'_tracklet']].rename({metric+'_tracklet': f'{fold}'}, axis=1) for fold in range(5)], axis=1)
            
            melted3 = pd.melt(comb_folds3.reset_index(), id_vars='epoch')
            sns.lineplot(data=melted3, x='epoch', y='value', label=f'validation {metric} by tracklet', ci='sd', color='r')
            plt.ylim((lower_limit, 1.15))
            plt.savefig(f'plots/{metric}__{dataset_name}__{pretrained}__image_vs_tracklet.png', dpi=300)
            plt.show()
            
# %%
for j, pretrained in enumerate(['untrained', 'ImageNet']):
    for metric in ['accuracy', 'f1', 'kappa']:
        for i, dataset_name in enumerate(datasets):
            comb_folds1 = pd.concat([blub[j, fold, i].loc[0:51, [metric]].rename({metric: f'{fold}'}, axis=1) for fold in range(5)], axis=1)
            
            melted1 = pd.melt(comb_folds1.reset_index(), id_vars='epoch')
            sns.lineplot(data=melted1, x='epoch', y='value', label=f'{dataset_name}', ci='sd')
        plt.ylabel(metric)
        lower_limit = 0.3 if metric != 'kappa' else -0.2
        plt.ylim((lower_limit, 1.15))
        plt.savefig(f'plots/comparison__{metric}__{pretrained}__lineplot.png', dpi=300)
        plt.show()
# %%
new_cols = [col for col in blub[0, 0, 0] if col.startswith('svm_mAP_1')]
print(new_cols)
# %%
for j, pretrained in enumerate(['untrained', 'ImageNet']):
    for meth in ['1nn', 'logreg', 'svm']:
        for metric in ['mAP', 'accuracy', 'kappa']:
            for gallery_size in [1, 2, 4, 8]:
                blub[j, 0, ] 
                
            