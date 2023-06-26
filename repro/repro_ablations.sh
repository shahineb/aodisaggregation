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



# ABLATION 1 : GP only, no idealised vertical profile
OUTDIR=experiments/data/outputs/ablation/svgp-gp-only/
for SEED in {1..20};
do
  DIRNAME=seed_$SEED
  python ablations/run_svgp_gp_only.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
done


# ABLATION 2 : no meteorological inputs, spatiotemporal inputs only
OUTDIR=experiments/data/outputs/ablation/svgp-spatiotemporal-only/
for SEED in {1..20};
do
  DIRNAME=seed_$SEED
  python ablations/run_svgp_spatiotemporal_only.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
done


# ABLATION 3 : no spatiotemporal inputs, meteorological inputs only
OUTDIR=experiments/data/outputs/ablation/svgp-meteorological-only/
for SEED in {1..20};
do
  DIRNAME=seed_$SEED
  python ablations/run_svgp_meteorological_only.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
done


# ABLATION 4 : product structure for kernel
OUTDIR=experiments/data/outputs/ablation/svgp-product-kernel/
for SEED in {1..20};
do
  DIRNAME=seed_$SEED
  python ablations/run_svgp_product_kernel.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
done


# ABLATION 5 : additive structure for kernel
OUTDIR=experiments/data/outputs/ablation/svgp-additive-kernel/
for SEED in {1..20};
do
  DIRNAME=seed_$SEED
  python ablations/run_svgp_additive_kernel.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
done


# # ABLATION 6 : joint Matern kernel
# OUTDIR=experiments/data/outputs/ablation/svgp-jointmatern-kernel/
# for SEED in {1..20};
# do
#   DIRNAME=seed_$SEED
#   python ablations/run_svgp_jointmatern_kernel.py --seed=$SEED --cfg=$SVGP_CFG --o=$OUTDIR/$DIRNAME --device=$DEVICE
# done





































#
