
# call what needs to be called first to set it up

bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main.py -FOLD_I 0 -KFOLDS 5 -model_backend resnet50 -train_batch 16
bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main.py -FOLD_I 1 -KFOLDS 5 -model_backend resnet50 -train_batch 16
bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main.py -FOLD_I 2 -KFOLDS 5 -model_backend resnet50 -train_batch 16
bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main.py -FOLD_I 3 -KFOLDS 5 -model_backend resnet50 -train_batch 16
bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main.py -FOLD_I 4 -KFOLDS 5 -model_backend resnet50 -train_batch 16


bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Ensemble -name May31EnsembleFullUnbalanced10It_AugOn -AL_iterations 10 -train_augmentation True
bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Random -name May31RandomFullUnbalanced10It_AugOn -AL_iterations 10 -train_augmentation True

bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Ensemble -name May31EnsembleFullUnbalanced20It_AugOn -AL_iterations 20 -train_augmentation True
bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Random -name May31RandomFullUnbalanced20It_AugOn -AL_iterations 20 -train_augmentation True

bsub -n 1 -W 48:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Ensemble -name May31EnsembleFullUnbalanced20It_Take2_AugOn -AL_iterations 20 -train_augmentation True

bsub -n 1 -W 24:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Ensemble -name May31EnsembleFullUnbalanced20It5Models -AL_iterations 20 -AL_Ensemble_numofmodels 5
bsub -n 1 -W 48:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Ensemble -name May31EnsembleFullUnbalanced20It5Models_Take2 -AL_iterations 20 -AL_Ensemble_numofmodels 5



bsub -n 1 -W 48:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Ensemble -name May31EnsembleFullUnbalanced10It_Take3 -AL_iterations 10
bsub -n 1 -W 48:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Random -name May31RandomFullUnbalanced10It_Take3 -AL_iterations 10
bsub -n 1 -W 48:00 -R "rusage[mem=32000, ngpus_excl_p=1]" python3 main_al.py -AL_method Ensemble -name May31EnsembleFullUnbalanced10It5Models -AL_iterations 10 -AL_Ensemble_numofmodels 5
