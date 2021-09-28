TEST_CONFIG_DIR=test/toy/config
TEST_OUTPUT_DIR=test/toy/outputs


python run_ridge_regression.py --cfg=$TEST_CONFIG_DIR/ridge_regression.yaml --o=$TEST_OUTPUT_DIR/ridge-regression
python run_two_stage_ridge_regression.py --cfg=$TEST_CONFIG_DIR/two_stage_ridge_regression.yaml --o=$TEST_OUTPUT_DIR/two-stage-ridge-regression
python run_warped_ridge_regression.py --cfg=$TEST_CONFIG_DIR/warped_ridge_regression.yaml --o=$TEST_OUTPUT_DIR/warped-ridge-regression
python run_warped_two_stage_ridge_regression.py --cfg=$TEST_CONFIG_DIR/warped_two_stage_ridge_regression.yaml --o=$TEST_OUTPUT_DIR/warped-two-stage-ridge-regression
python run_kernel_ridge_regression.py --cfg=$TEST_CONFIG_DIR/kernel_ridge_regression.yaml --o=$TEST_OUTPUT_DIR/kernel-ridge-regression
python run_two_stage_kernel_ridge_regression.py --cfg=$TEST_CONFIG_DIR/two_stage_kernel_ridge_regression.yaml --o=$TEST_OUTPUT_DIR/two-stage-kernel-ridge-regression
python run_warped_kernel_ridge_regression.py --cfg=$TEST_CONFIG_DIR/warped_kernel_ridge_regression.yaml --o=$TEST_OUTPUT_DIR/warped-kernel-ridge-regression
python run_warped_two_stage_kernel_ridge_regression.py --cfg=$TEST_CONFIG_DIR/warped_two_stage_kernel_ridge_regression.yaml --o=$TEST_OUTPUT_DIR/warped-two-stage-kernel-ridge-regression
