# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from reid.metric_logging import load_run_data, load_test_data
from reid.utils import mkdir_if_missing
import glob
import os
import traceback

sns.set(style='ticks')
sns.set_style('whitegrid')

def perm_tuple(x, indices):
    return tuple(x[i] for i in indices)

# %%
datasets = ['2_NBapril_pad', '4_berlin_pad', '6_vienna_pad', '8_mulhouse_pad']

dataset_colors = ['#ff8800', '#0055dd', '#00cc00', '#ea57c5']

def scenario_translation(x):
    result = ''
    result += f'{"multi" if x%2 == 0 else "single"}_train__'
    x >>= 1
    result += f'{"full_network" if x%2 == 0 else "frozen_trunk"}__'
    x >>= 1
    result += f'{"multi" if x%2 == 0 else "single"}_head'
    
    return result

def get_complement_sets(ds):
    return [s for s in datasets if s not in ds]

def prepare_figure(id=0):
    fig = plt.figure(id)
    ax = fig.add_subplot(111)
    
    return fig, ax

def legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -0.2), borderaxespad=0)
    return lgd
 
def save_fig(fig, path, lgd):
    if lgd is not None:
        fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    else:
        plt.legend([], [], frameon=False)
        fig.savefig(path, bbox_inches='tight', dpi=300)

# %%
blub = np.zeros((5, 2, 5, len(datasets)), dtype=object)
blubtrain = np.zeros((5, 2, 5, len(datasets)), dtype=object)
confusion = np.zeros((2, len(datasets), 5, 6, 6), dtype=int)

for run in glob.glob('logs/Run_*'):
    dm, tm, ts, ms, pt, sh, ft = load_run_data(run)
    scenario = 0b000
    if len(ts) == 1:
        scenario |= 0b001
    if ft:
        scenario |= 0b010
    if sh:
        scenario |= 0b100
        
    print(f'{len(ts)=} {ft=} {sh=} {scenario=}')
    print(scenario_translation(scenario))
        
    pt = 1 if pt else 0
    
    data = {}
    fold = None
    for dataset_name, retrieved_fold in ts:
        if fold == None:
            fold = retrieved_fold
        else:
            assert fold == retrieved_fold
        print(dataset_name, fold, pt)
        data[dataset_name] = pd.read_csv(dm[dataset_name])
        data[dataset_name].set_index('epoch', inplace=True)
        data[dataset_name] = data[dataset_name][~data[dataset_name].index.duplicated(keep='first')]
        data[dataset_name] = data[dataset_name].loc[:, ~data[dataset_name].columns.str.replace("(\.\d+)$", "").duplicated()]
        
    for dataset_name, retrieved_fold in ms:
        assert fold == retrieved_fold
        data[dataset_name] = pd.read_csv(dm[dataset_name])
        data[dataset_name].set_index('epoch', inplace=True)
        data[dataset_name] = data[dataset_name][~data[dataset_name].index.duplicated(keep='first')]
        data[dataset_name] = data[dataset_name].loc[:, ~data[dataset_name].columns.str.replace("(\.\d+)$", "").duplicated()]
        
    if len(ts) == 1:
        dataset_name = ts[0][0]
    else:
        ts = [dataset_name for dataset_name, _ in ts]
        for dataset_name in datasets:
            if dataset_name not in ts:
                break
    blub[scenario, pt, int(fold)-1, datasets.index(dataset_name)] = data
    
    
    tm.set_index('epoch', inplace=True)
    data = tm[~tm.index.duplicated(keep='first')]
    data = data.loc[:, ~data.columns.str.replace("(\.\d+)$", "").duplicated()]
    
    if sh:
        confusion[pt, datasets.index(dataset_name), int(fold)-1] = np.load(os.path.join(run, 'confusion.npy'))
        
    blubtrain[scenario, pt, int(fold)-1, datasets.index(dataset_name)] = data

np.save('confusion.npy', confusion)

# %%
iterables = (
    range(0b101),
    ('untrained', 'ImageNet'),
    ('1nn', 'logreg', 'svm'),
    ('accuracy', 'f1', 'kappa', 'mAP'),
    (1, 2, 4, 8, 16),
    range(5),
    range(10),
    datasets,
    datasets
)

idx = pd.MultiIndex.from_product(iterables, names=(
    'scenario', 'pretrained', 'gallery_training', 'metric', 'fold_1', 'fold_2', 'gallery_size', 'source_set', 'monitor_set'
))

df_transfer_during_training = pd.DataFrame(
    np.zeros((50, np.prod([len(it) for it in iterables]))),
    columns=idx, index=[i for i in range(50)]
)

iterables = (
    range(0b101),
    ('untrained', 'ImageNet'),
    ('accuracy', 'f1', 'kappa'),
    range(5),
    datasets
)

idx = pd.MultiIndex.from_product(iterables, names=(
    'scenario', 'pretrained', 'metric', 'fold', 'dataset'
))

df_supervised_during_training_single_train = pd.DataFrame(
    np.zeros((50, np.prod([len(it) for it in iterables]))),
    columns=idx, index=[i for i in range(50)]
)

iterables = (
    range(0b101),
    ('untrained', 'ImageNet'),
    ('accuracy', 'f1', 'kappa'),
    range(5),
    datasets,
    datasets
)

idx = pd.MultiIndex.from_product(iterables, names=(
    'scenario', 'pretrained', 'metric', 'fold', 'target_dataset', 'train_dataset'
))

df_supervised_during_training_multi_train = pd.DataFrame(
    np.zeros((50, np.prod([len(it) for it in iterables]))),
    columns=idx, index=[i for i in range(50)]
)

# %%
def clear_plt():
    plt.close()
    plt.cla()
    plt.clf()
plt.show = clear_plt

BUILD_BIG_DATAFRAME = False
NO_PLOT_OVERRIDE = True
BASE_LINEPLOT = True and not NO_PLOT_OVERRIDE
BASE_BOXPLOT = True and not NO_PLOT_OVERRIDE
GALLERY_SIZE_PLOTS = True and not NO_PLOT_OVERRIDE
DATASET_COMP_PLOTS = True and not NO_PLOT_OVERRIDE
TRACKLET_COMP_PLOTS = True and not NO_PLOT_OVERRIDE
SCATTER_PLOTS = True and not NO_PLOT_OVERRIDE
PLOT_LEGENDS = False

for scenario in range(0b101):
    save_dir = os.path.join('plots-no-legend', scenario_translation(scenario))
    mkdir_if_missing(save_dir)


    corr_coefficients = np.zeros((2, 3, 4, 2, len(datasets), len(datasets)))

    multi_train = not scenario&1
    for i, dataset_name in enumerate(datasets):
        
        if multi_train:
            monitor_sets = [dataset_name]
            train_sets = get_complement_sets([dataset_name])
        else:
            monitor_sets = get_complement_sets([dataset_name])
            train_sets = [dataset_name]
        
        for j, pretrained in enumerate(['untrained', 'ImageNet']):
            if BUILD_BIG_DATAFRAME:
                df_transfer_during_training = df_transfer_during_training.copy()
            for metric in ['accuracy', 'f1', 'kappa', 'mAP']:
                print(f'{dataset_name} {pretrained} {metric}:')
                for fold in range(5):
                    print(f'  fold {fold}')
                    if metric != 'mAP':
                        if multi_train:
                            for train_set in train_sets:
                                df_supervised_during_training_multi_train.loc[:, (scenario, pretrained, metric, fold, dataset_name, train_set)] = \
                                    blub[scenario, j, fold, i][train_set].loc[:50, metric]
                        else:
                            df_supervised_during_training_single_train.loc[:, (scenario, pretrained, metric, fold, dataset_name)] = \
                                blub[scenario, j, fold, i][dataset_name].loc[:50, metric]
                    if BUILD_BIG_DATAFRAME:
                        for monitor_set in datasets:
                            for gallery_training in ['1nn', 'logreg', 'svm']:
                                for gallery_size in [1, 2, 4, 8]:
                                    if monitor_set == '8_mulhouse_pad' and gallery_size == 8:
                                        continue
                                    for fold_2 in range(10):
                                        print(f'       {dataset_name} {pretrained} {metric}: {monitor_set} {gallery_training} {gallery_size} {fold_2}')
                                        df_transfer_during_training.loc[
                                            :50, (
                                                scenario,
                                                pretrained,
                                                gallery_training,
                                                metric,
                                                fold,
                                                fold_2,
                                                gallery_size,
                                                dataset_name,
                                                monitor_set
                                            )
                                        ] = blub[scenario, j, fold, i][monitor_set].loc[:50, f'{gallery_training}_{metric}_{gallery_size}_{fold_2}']
                                        
                fold = 0
                print(type(blub[scenario, j, fold, i]))
                if metric == 'mAP':
                    continue
                
                comb_folds1 = pd.concat([
                    blub[scenario, j, fold, i][dataset].loc[0:50, [metric]].rename({metric: f'{fold}_{dataset}'}, axis=1)
                        for fold in range(5)
                        for dataset in train_sets
                ], axis=1)
                
                melted1 = pd.melt(comb_folds1.reset_index(), id_vars='epoch')
                
                if BASE_LINEPLOT:
                    fig, ax = prepare_figure()
                    sns.lineplot(data=melted1, x='epoch', y='value', label=f'validation {metric}', ci=95, ax=ax)
                    
                    if metric == 'accuracy':
                        comb_folds2 = pd.concat([
                            blubtrain[scenario, j, fold, i].loc[0:50, ['accuracy']].rename({'accuracy': f'{fold}'}, axis=1)
                                for fold in range(5)
                        ], axis=1)
                        
                        melted2 = pd.melt(comb_folds2.reset_index(), id_vars='epoch')
                        
                        sns.lineplot(data=melted2, x='epoch', y='value', label=f'training accuracy', ci=95, color='#cc4488', ax=ax)
                    
                    
                    lower_limit = 0.3
                    if metric == 'kappa':
                        lower_limit = -0.2
                    elif (scenario>>2)&1:
                        lower_limit = -0.05
                    
                    ax.set_ylim((lower_limit, 1.15))
                    ldg = legend(ax) if PLOT_LEGENDS else None
                    save_fig(fig, os.path.join(save_dir, f'{metric}__{dataset_name}__{pretrained}__lineplot.png'), ldg)
                    plt.show()
                
                comb_folds1_mean = comb_folds1.mean(axis=1)
                
                if GALLERY_SIZE_PLOTS:
                    for dataset in monitor_sets:
                        for gallery_training in ['1nn', 'logreg', 'svm']:
                            fig0, ax0 = prepare_figure(0)
                            sns.lineplot(data=comb_folds1_mean, label=f'validation {metric}', ax=ax0)
                            for gallery_size in [1, 2, 4, 8]:
                                if dataset == '8_mulhouse_pad' and gallery_size == 8:
                                    continue
                                
                                comb_folds = pd.concat([
                                    blub[scenario, j, fold_1, i][dataset]
                                        .loc[0:50, [f'{gallery_training}_{metric}_{gallery_size}_{fold_2}' for fold_2 in range(10)]]
                                        .rename({f'{gallery_training}_{metric}_{gallery_size}_{fold_2}': f'{fold_1}_{fold_2}' for fold_2 in range(10)}, axis=1)
                                            for fold_1 in range(5)
                                ], axis=1)
                                melted = pd.melt(comb_folds.reset_index(), id_vars='epoch')
                                color = f'#{0xaaccaa - int(np.log2(gallery_size)*0x330033):06x}'
                                sns.lineplot(data=melted, x='epoch', y='value', label=f'{metric} {gallery_size=}', ci=95, color=color, dashes=True, ax=ax0)
                                
                            ax0.set_ylim((lower_limit+0.1, 1.00))
                            ldg0 = legend(ax0) if PLOT_LEGENDS else None
                            save_fig(fig0, os.path.join(save_dir, f'{metric}__{",".join(train_sets)}->{dataset}__{pretrained}__{gallery_training}__lineplot.png'), ldg0)
                            plt.show()
                
                if DATASET_COMP_PLOTS:
                    if multi_train:
                        fig, ax = prepare_figure()
                        for k, dataset in enumerate(train_sets):
                            comb_folds = pd.concat([
                                blub[scenario, j, fold, i][dataset].loc[0:50, [metric]].rename({metric: f'{fold}_{dataset}'}, axis=1)
                                    for fold in range(5)
                            ], axis=1)
                    
                            melted1 = pd.melt(comb_folds.reset_index(), id_vars='epoch')
                            sns.lineplot(data=melted1, x='epoch', y='value', label=f'{dataset} {metric}', ci=95, ax=ax, color=dataset_colors[datasets.index(dataset)])
                            
                        ax.set_ylim((lower_limit, 1.15))
                        ldg = legend(ax) if PLOT_LEGENDS else None
                        save_fig(fig, os.path.join(save_dir, f'{metric}__{dataset_name}__{pretrained}__comparison__lineplot.png'), ldg)
                        plt.show()
                
                if BASE_BOXPLOT:
                    sns.boxplot(data=comb_folds1.loc[::5].T, color='#6699ee')
                    plt.ylim((lower_limit, 1.15))
                    plt.savefig(os.path.join(save_dir, f'{metric}__{dataset_name}__{pretrained}__boxplot.png'), dpi=300)
                    plt.show()
                
                if TRACKLET_COMP_PLOTS:
                    fig, ax = prepare_figure()
                    sns.lineplot(data=melted1, x='epoch', y='value', label=f'validation {metric}', ci=95, ax=ax)
                    comb_folds3 = pd.concat([
                        blub[scenario, j, fold, i][dataset_name].loc[0:50, [metric+'_tracklet']].rename({metric+'_tracklet': f'{fold}'}, axis=1)
                            for fold in range(5)
                            for dataset_name in train_sets 
                    ], axis=1)
                    
                    melted3 = pd.melt(comb_folds3.reset_index(), id_vars='epoch')
                    sns.lineplot(data=melted3, x='epoch', y='value', label=f'validation {metric} by tracklet', ci=95, color='r', ax=ax)
                    ax.set_ylim((lower_limit, 1.15))
                    ldg = legend(ax) if PLOT_LEGENDS else None
                    save_fig(fig, os.path.join(save_dir, f'{metric}__{dataset_name}__{pretrained}__image_vs_tracklet.png'), ldg)
                    plt.show()
                
                for k, dataset in enumerate(datasets):
                    for l, gallery_size in enumerate([1, 2, 4, 8]):
                        if gallery_size == 4 and SCATTER_PLOTS:
                            fig, ax = prepare_figure()
                        
                        for m, gallery_training in enumerate(['1nn', 'logreg', 'svm']):
                            if dataset == '8_mulhouse_pad' and gallery_size == 8:
                                continue
                            
                            to_comb = [
                                blub[scenario, j, fold_1, i][dataset]
                                    .loc[0:50, [f'{gallery_training}_{metric}_{gallery_size}_{fold_2}' for fold_2 in range(10)]]
                                    .rename({f'{gallery_training}_{metric}_{gallery_size}_{fold_2}': f'{fold_1}_{fold_2}' for fold_2 in range(10)}, axis=1)
                                        for fold_1 in range(5)
                            ]
                            
                            comb_folds_mean = pd.concat(to_comb, axis=1).mean(axis=1)
                            
                            if metric == 'kappa':
                                corr_coefficients[j, m, l, 0, i, k] = np.corrcoef(comb_folds_mean, comb_folds1_mean)[0, 1]
                                corr_coefficients[j, m, l, 1, i, k] = np.corrcoef(comb_folds_mean[1:], comb_folds1_mean[1:])[0, 1]
                            
                            if gallery_size == 4 and SCATTER_PLOTS:
                                sns.scatterplot(comb_folds1_mean, comb_folds_mean, label=f'{gallery_training}')
                        if gallery_size == 4 and SCATTER_PLOTS:
                            ldg = legend(ax) if PLOT_LEGENDS else None
                            save_fig(fig, os.path.join(save_dir, f'{metric}__{dataset_name}__{pretrained}__scatter__->{dataset}.png'), ldg)
                            plt.show()
                        
    if multi_train:
        ylabels = [',\n'.join(get_complement_sets(dataset)) for dataset in datasets]
    else:
        ylabels = datasets

    figsize = (8, 6)
    plt.figure(0, figsize=figsize)  
    ax = sns.heatmap(
        corr_coefficients[1, :, :3, 0].mean(axis=(0, 1)),
        annot=True,
        cmap='viridis',
        vmin=-1, vmax=1,
        xticklabels=datasets, yticklabels=ylabels
    )
    ax.figure.tight_layout()
    plt.savefig(os.path.join(save_dir, f'correlation_matrix__mean__ImageNet.png'), dpi=300)
    plt.show()

    plt.figure(0, figsize=figsize)  
    ax = sns.heatmap(
        corr_coefficients[0, :, :3, 0].mean(axis=(0, 1)),
        annot=True,
        cmap='viridis',
        vmin=-1, vmax=1,
        xticklabels=datasets, yticklabels=ylabels
    )
    ax.figure.tight_layout()
    plt.savefig(os.path.join(save_dir, f'correlation_matrix__mean__untrained.png'), dpi=300)
    plt.show()

    plt.figure(0, figsize=figsize)  
    ax = sns.heatmap(
        corr_coefficients[1, :, :3, 0].std(axis=(0, 1)),
        annot=True,
        cmap='plasma',
        vmin=0, vmax=0.5,
        xticklabels=datasets, yticklabels=ylabels
    )
    ax.figure.tight_layout()
    plt.savefig(os.path.join(save_dir, f'correlation_matrix__std__ImageNet.png'), dpi=300)
    plt.show()

    plt.figure(0, figsize=figsize)  
    ax = sns.heatmap(
        corr_coefficients[0, :, :3, 0].std(axis=(0, 1)),
        annot=True,
        cmap='plasma',
        vmin=0, vmax=0.5,
        xticklabels=datasets, yticklabels=ylabels
    )
    ax.figure.tight_layout()
    plt.savefig(os.path.join(save_dir, f'correlation_matrix__std__untrained.png'), dpi=300)
    plt.show()

df_supervised_during_training_multi_train.to_csv('supervised_stats_during_training_multi_train.csv')
df_supervised_during_training_single_train.to_csv('supervised_stats_during_training_single_train.csv')
if BUILD_BIG_DATAFRAME:
    df_transfer_during_training.to_csv('transfer_stats_during_training.csv')
corr_coefficients.shape

corr_coefficients[0, :, :3, 0].mean(axis=(0, 1))
sns.heatmap(
    corr_coefficients[1, :, :3, 0].mean(axis=(0, 1)),
    annot=True,
    cmap='viridis',
    vmin=-1, vmax=1,
    xticklabels=datasets, yticklabels=['1nn', 'logreg', 'svm']
)

# %%
sns.heatmap(corr_coefficients[1, :, :3, 0].std(axis=(0, 1)), annot=True, cmap='viridis')

# %%
corr_coefficients[1, :, :3, 0].std(axis=(0, 1))

# %%
print(comb_folds1.iloc[1:51].mean(axis=1), comb_folds.iloc[1:51].mean(axis=1))
sns.scatterplot(comb_folds1.iloc[1:51].mean(axis=1), comb_folds.iloc[1:51].mean(axis=1))
np.corrcoef(comb_folds1.iloc[1:51].mean(axis=1), comb_folds.iloc[1:51].mean(axis=1))

# %%

blubtest = np.zeros((5, 2, 5, len(datasets)), dtype=object)

all_test_data = []

for test in glob.glob('logs/Test_*'):
    dm, ts, ms, pt, sh, ft = load_test_data(test)
    scenario = 0b000
    if len(ts) == 1:
        scenario |= 0b001
    if ft:
        scenario |= 0b010
    if sh:
        scenario |= 0b100
    
    print(test)
    print(f'{len(ts)=} {ft=} {sh=} {scenario=}')
    print(scenario_translation(scenario))
        
    pt = 1 if pt else 0
    
    data = {}
    fold = None
        
    for dataset_name, retrieved_fold in ms:
        retrieved_fold = retrieved_fold.split('~')[-1]
        if fold is None:
            fold = retrieved_fold
        else:
            assert fold == retrieved_fold
        data[dataset_name] = pd.read_csv(dm[dataset_name])
        data[dataset_name].set_index('epoch', inplace=True)
        data[dataset_name] = data[dataset_name][~data[dataset_name].index.duplicated(keep='first')]
        data[dataset_name] = data[dataset_name].loc[:, ~data[dataset_name].columns.str.replace("(\.\d+)$", "").duplicated()]
        
    if len(ts) == 1:
        dataset_name = ts[0][0]
    else:
        ts = [dataset_name for dataset_name, _ in ts]
        for dataset_name in datasets:
            if dataset_name not in ts:
                break
    blubtest[scenario, pt, int(fold.split('~')[-1])-1, datasets.index(dataset_name)] = data
    

# %%
iterables = (range(0b101), ('untrained', 'ImageNet'), ('1nn', 'logreg', 'svm'), ('accuracy', 'f1', 'kappa', 'mAP'), (1, 2, 4, 8, 16), datasets, datasets)
idx = pd.MultiIndex.from_product(iterables, names=('scenario', 'pretrained', 'gallery_training', 'metric', 'gallery_size', 'source_set', 'monitor_set'))

df = pd.DataFrame(np.zeros((50, np.prod([len(it) for it in iterables]))), columns=idx, index=[f'{fold1}_{fold2}' for fold1 in range(5) for fold2 in range(10)])

# %%
for scenario in range(0b101):
    multi_train = not scenario&1

    for i, dataset_name in enumerate(datasets):
        if multi_train:
            monitor_sets = [dataset_name]
            train_sets = get_complement_sets([dataset_name])
        else:
            monitor_sets = get_complement_sets([dataset_name])
            train_sets = [dataset_name]
            
        for j, pretrained in enumerate(['untrained', 'ImageNet']):
            for k, metric in enumerate(['accuracy', 'f1', 'kappa', 'mAP']):
                for m, gallery_training in enumerate(['1nn', 'logreg', 'svm']):
                    for n, gallery_size in enumerate([1, 2, 4, 8, 16]):
                        for l, monitor_set in enumerate(datasets):                            
                            to_comb = []
                            try:
                                for fold_1 in range(5):
                                    blubblub = blubtest[scenario, j, fold_1, i]
                                    print(type(blubblub), blubblub.keys())
                                    to_comb.append(
                                        blubblub[monitor_set]
                                        .loc[0, [f'{gallery_training}_{metric}_{gallery_size}_{fold_2}' for fold_2 in range(10)]]
                                        .rename({f'{gallery_training}_{metric}_{gallery_size}_{fold_2}': f'{fold_1}_{fold_2}' for fold_2 in range(10)}, axis=1)
                                    )
                            except KeyError as e:
                                print(f'KeyError: {scenario=} {dataset_name=} {pretrained=} {metric=} {gallery_training=} {gallery_size=} {monitor_set=}')
                                continue
                            except TypeError as e:
                                traceback.print_exc()
                                print(f'TypeError: {scenario=} {dataset_name=} {pretrained=} {metric=} {gallery_training=} {gallery_size=} {monitor_set=}')
                                continue
                                
                                
                            comb_folds = pd.concat(to_comb, axis=0)
                            df[scenario, pretrained, gallery_training, metric, gallery_size, monitor_set, dataset_name] = comb_folds
            

# %%
df.to_csv('transfer_test_stats.csv')
df.loc[:, 0b001].mean()

# %%

df.loc[:, 0b001]

# %% 
dataset_translation = {
    '2_NBapril_pad': 'Nuremberg',
    '4_berlin_pad': 'Berlin',
    '6_vienna_pad': 'Vienna',
    '8_mulhouse_pad': 'Mulhouse'
}
dttr = lambda x: dataset_translation[x]

# %%
save_dir ='plots'
scenario = 0b001
multi_train = not scenario&1
target = (scenario, 'ImageNet', 'logreg', 'accuracy', 1)

app_data = df.loc[:, target].mean().to_frame().unstack()
app_std = df.loc[:, target].std().to_frame().unstack()
if multi_train:
    ylabels = [',\n'.join(map(dttr, get_complement_sets(dataset))) for dataset in datasets]
else:
    ylabels = datasets
xlabels = list(map(dttr, datasets))
# %%
final_data = app_data.applymap(lambda x: f'{x:.2f} ± ') + app_std.applymap(lambda x: f'{x:.2f}') #±
plt.figure(figsize=(8, 6))
ax = sns.heatmap(app_data, cmap='viridis', annot=final_data, yticklabels=ylabels, xticklabels=xlabels, vmin=0, vmax=1, fmt='')
plt.yticks(rotation=0)
ax.figure.tight_layout()
ax.set(xlabel='target set', ylabel='train set')
plt.savefig(os.path.join(save_dir, f'{"__".join(map(str, target))}__transfer_heatmap.png'))
plt.show()
#app_std.applymap(lambda x: f'{x:.2f}') 
# %%
np.array(final_data)
# %%
