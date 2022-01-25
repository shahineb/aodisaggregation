# Define configuration files path variables
IDEAL_CFG=config/ideal.yaml


# Define output directories path variables
IDEAL_OUTDIR=experiments/data/outputs/ideal


# Run experiments for multiple seeds
python run_ideal_vertical_profile.py --seed=$SEED --cfg=$IDEAL_CFG --o=$IDEAL_OUTDIR --plot
