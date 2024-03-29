import os
import stat
import numpy as np
from reid import metric_logging as logging
from reid.utils import mkdir_if_missing
import glob
import fcntl
import shutil

datasets = ['2_NBapril_pad', '4_berlin_pad', '6_vienna_pad', '8_mulhouse_pad']

def write_script(train_config: str, model_config: str, datasets: str, monitor_sets: str, dest: str):
    with open(dest, 'w') as f:
        f.write(
            f"#!/bin/bash -l\n"
            f"#PBS -l nodes=1:ppn=8:cuda10,walltime=04:00:00\n"
            f"#PBS -o /home/hpc/iwso/iwso035h/polarbear-reid/train_scripts/experiment_logs\n"
            f"#PBS -e /home/hpc/iwso/iwso035h/polarbear-reid/train_scripts/experiment_logs\n\n"
            f"source ~/torch-cuda\n\n"
            f"cd /home/hpc/iwso/iwso035h/polarbear-reid\n"
            f"python train.py \\\n"
            f"    --output-to-file \\\n"
            f"    --data-dir data/icebears \\\n"
            f"    --datasets {datasets} \\\n"
            f"    --monitor-datasets {monitor_sets} \\\n"
            f"    --training-config {train_config} \\\n"
            f"    --model-config {model_config} \\\n"
            f"    --gpu-devices 0,1"
        )
        
def write_resume_script(resume_config: str, dest: str):
    with open(dest, 'w') as f:
        f.write(
            f"#!/bin/bash -l\n"
            f"#PBS -l nodes=1:ppn=8:cuda10,walltime=04:00:00\n"
            f"#PBS -o /home/hpc/iwso/iwso035h/polarbear-reid/train_scripts/experiment_logs\n"
            f"#PBS -e /home/hpc/iwso/iwso035h/polarbear-reid/train_scripts/experiment_logs\n\n"
            f"source ~/torch-cuda\n\n"
            f"cd /home/hpc/iwso/iwso035h/polarbear-reid\n"
            f"python train.py \\\n"
            f"    --output-to-file \\\n"
            f"    --data-dir data/icebears \\\n"
            f"    --resume-config {resume_config} \\\n"
            f"    --gpu-devices 0,1"
        )

def write_test_script(resume_config: str, datasets: str, dest: str):
    with open(dest, 'w') as f:
        f.write(
            f"#!/bin/bash -l\n"
            f"#PBS -l nodes=1:ppn=8:cuda10,walltime=00:30:00\n"
            f"#PBS -o /home/hpc/iwso/iwso035h/polarbear-reid/train_scripts/experiment_logs\n"
            f"#PBS -e /home/hpc/iwso/iwso035h/polarbear-reid/train_scripts/experiment_logs\n\n"
            f"source ~/torch-cuda\n\n"
            f"cd /home/hpc/iwso/iwso035h/polarbear-reid\n"
            f"python test.py \\\n"
            f"    --output-to-file \\\n"
            f"    --data-dir data/icebears \\\n"
            f"    --datasets {datasets} \\\n"
            f"    --resume-config {resume_config} \\\n"
            f"    --gpu-devices 0,1"
        )

mkdir_if_missing('train_scripts/experiments')
for script in glob.glob('train_scripts/experiments/*.sh'):
    os.remove(script)

blub = np.zeros((2, 5, len(datasets), 3), dtype=bool)
blubfrozen = np.zeros((2, 5, len(datasets), 3), dtype=bool)

blubtest = np.zeros((2, 5, len(datasets), 3), dtype=bool)
blubtestfrozen = np.zeros((2, 5, len(datasets), 3), dtype=bool)
    
for tgt in sorted(glob.glob(os.path.join('logs', 'Test_*')), key=lambda x: int(x.split('_')[-1])):
    dm, ts, ms, pre, sh, ft = logging.load_test_data(tgt)
    pre = 1 if pre else 0
    if len(ts) == 1:
        ss = datasets.index(ts[0][0])
        sh = 0
    else:
        ss = datasets.index(ms[0][0])
        if sh and not ft:
            sh = 2
        else:
            sh = 1
    
    print(pre, int(ts[0][1])-1, ss, sh)
    if ft:
        blubtestfrozen[pre, int(ts[0][1])-1, ss, sh] = True
    else:
        blubtest[pre, int(ts[0][1])-1, ss, sh] = True

for tgt in sorted(glob.glob(os.path.join('logs', 'Run_*')), key=lambda x: int(x.split('_')[-1])):
    dm, tm, ts, ms, pre, sh, ft = logging.load_run_data(tgt)
    print()
    pre = 1 if pre else 0
    if len(ts) == 1:
        ss = datasets.index(ts[0][0])
        sh = 0
    else:
        ss = datasets.index(ms[0][0])
        if sh:
            sh = 2
        else:
            sh = 1
    print(np.max(tm.loc[:, 'epoch']))
    print(pre, int(ts[0][1])-1, ss, sh)
    if ft:
        if blubfrozen[pre, int(ts[0][1])-1, ss, sh]:
            print('Alert!!')
            #shutil.rmtree(tgt)
            continue
        blubfrozen[pre, int(ts[0][1])-1, ss, sh] = True
    else:
        if blub[pre, int(ts[0][1])-1, ss, sh]:
            print('Alert!!')
            #shutil.rmtree(tgt)
            continue
        blub[pre, int(ts[0][1])-1, ss, sh] = True
        
    print(tgt)
    with open(os.path.join(tgt, '.lock'), 'w') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            not_training = True
        except BlockingIOError as e:
            not_training = False
    print(np.max(tm.loc[:, 'epoch']) < 50, not_training)
    if not_training:
        if np.max(tm.loc[:, 'epoch']) < 50:
            write_resume_script(
                f'{tgt}/checkpoint/resume.yaml',
                f'train_scripts/experiments/resume_{tgt.split("/")[-1]}.sh'
            )
        elif ft:
            if not blubtestfrozen[pre, int(ts[0][1])-1, ss, sh]:
                write_test_script(
                    f'{tgt}/checkpoint/resume.yaml',
                    " ".join([f'{ds}:~{spl}' for ds, spl in ms] + [f'{ds}:{spl}' for ds, spl in ts]),
                    f'train_scripts/experiments/test_{tgt.split("/")[-1]}.sh'
                )
        else:
            if not blubtest[pre, int(ts[0][1])-1, ss, sh]:
                write_test_script(
                    f'{tgt}/checkpoint/resume.yaml',
                    " ".join([f'{ds}:~{spl}' for ds, spl in ms] + [f'{ds}:{spl}' for ds, spl in ts]),
                    f'train_scripts/experiments/test_{tgt.split("/")[-1]}.sh'
                )
    
        
print(blub.sum())

for i, pretrained in enumerate(['untrained', 'ImageNet']):
    for j in range(5):
        for k, dataset in enumerate(datasets):
            sets = [f'{dataset}:{j+1}']
            c_sets = [f'{cset}:{j+1}' for cset in datasets if cset != dataset]
            model_config = f'configs/model/res50_{pretrained}_bn_512.yaml'
            train_config = 'configs/training/multi_head_isolated_triplet.yaml'
            
            if not blub[i, j, k, 0]:
                dest = f'train_scripts/experiments/{"_".join(sets)}__{pretrained}__multi_head.sh'
                write_script(train_config, model_config, " ".join(sets), " ".join(c_sets), dest)
            
            if not blub[i, j, k, 1]:
                dest = f'train_scripts/experiments/{"_".join(c_sets)}__{pretrained}__multi_head.sh'
                write_script(train_config, model_config, " ".join(c_sets), " ".join(sets), dest)
                
            if not blub[i, j, k, 2]:
                dest = f'train_scripts/experiments/{"_".join(c_sets)}__{pretrained}__single_head.sh'
                train_config = 'configs/training/single_head_isolated_triplet.yaml'
                write_script(train_config, model_config, " ".join(c_sets), " ".join(sets), dest)
                
            train_config = 'configs/training/multi_head_isolated_triplet_frozen_trunk.yaml'
            if not blubfrozen[i, j, k, 0]:
                dest = f'train_scripts/experiments/{"_".join(sets)}__{pretrained}__multi_head__frozen_trunk.sh'
                write_script(train_config, model_config, " ".join(sets), " ".join(c_sets), dest)
                
            if not blubfrozen[i, j, k, 1]:
                dest = f'train_scripts/experiments/{"_".join(c_sets)}__{pretrained}__multi_head__frozen_trunk.sh'
                write_script(train_config, model_config, " ".join(c_sets), " ".join(sets), dest)
            
with open('start_experiments.sh', 'w') as f:
    f.write(
        "#!/bin/bash -l\n"
        "find train_scripts/experiments -name '*.sh' | xargs -I {} qsub.tinygpu {}"
    )
        
st = os.stat('start_experiments.sh')
os.chmod('start_experiments.sh', st.st_mode | stat.S_IEXEC)

