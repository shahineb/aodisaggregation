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



# ABLATION : no pressure
OUTDIR=experiments/data/outputs/ablation/svgp-no-P/
for SEED in {1..10};
do
  DIRNAME=seed_$SEED
  python ablations/run_svgp_no_P.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
done


# ABLATION 2 : no temperature no pressure
OUTDIR=experiments/data/outputs/ablation/svgp-no-T-no-P/
for SEED in {1..10};
do
  DIRNAME=seed_$SEED
  python ablations/run_svgp_no_T_no_P.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
done































#
