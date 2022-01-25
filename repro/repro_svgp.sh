# Retrive device id
for i in "$@"
do
case $i in
    --device=*)
    DEVICE="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Define configuration files path variables
SVGP_CFG=config/svgp.yaml


# Define output directories path variables
SVGP_OUTDIR=experiments/data/outputs/svgp


# Run experiments for multiple seeds
for SEED in 2 3 5 7 11 ;
do
  DIRNAME=seed_$SEED
  python run_svgp_vertical_profile.py --seed=$SEED --cfg=$SVGP_CFG --o=$SVGP_OUTDIR/$DIRNAME --device=$DEVICE
done
