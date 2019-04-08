import json
import os
import glob
# from pandas.io.json import json_normalize


job_definition_file = f'../../baselines.json'
job_definition = json.loads(open(job_definition_file, 'r').read())

report_folder = f'/home/setepenre/mlperf_output/'


# bench_metrics = {
#     'natural_language_processing_word_language_model_pytorch_main.py': ['train_item', 'eval', 'all'],
#     'reinforcement_atari_pytorch_main.py': ['train_item'],
#     'generative_adversarial_networks_dcgan_pytorch_main.py': ['train_item'],
#     'image_classification_convnets_pytorch_conv_simple.py': ['train_item', 'loading', 'compute'],
#     'reinforcement_cart_pole_pytorch_reinforce.py': ['train_item'],
#     'image_classification_convnets_pytorch_conv_distributed.py': ['train_item', 'loading', 'compute'],
#     'super_resolution_subpixel_convolution_pytorch_main.py': ['train_item'],
#     'natural_language_processing_rnn_translator_pytorch_train_item.py': ['train_item'],
#     'regression_polynome_pytorch_main.py': ['train_item'],
#     'time_sequence_prediction_lstm_pytorch_train_item.py': ['train_item'],
#     'object_detection_single_stage_detector_pytorch_train_item.py': ['train_item'],
#     'variational_auto_encoder_auto_encoding_variational_bayes_pytorch_main.py': ['train_item'],
#     'fast_neural_style_neural_style_pytorch_neural_style.py': ['train_item'],
#     'natural_language_processing_word_language_model_pytorch_main_fp16_optimizer.py': ['train_item', 'eval', 'all'],
#     'image_loading_loader_pytorch_loaders.py': ['train_item'],
#     'recommendation_neural_collaborative_filtering_pytorch_ncf.py': ['loading_data', 'train_item']
# }


def score(report):
    values = report['train_item']
    avg, range = values['avg'], values['range']
    # min, max = values['min'], values['max']

    return avg / range


def report():
    filename = os.environ.get('REPORT_PATH', '/home/user1/mlperf/output/bench_result.csv')

    report_file = open(filename, 'r')
    reports = report_file.read()[:-1] + ']'
    report_file.close()

    for report in reports:
        print(json.dumps(report, indent=4))


def read_results(report_name):
    return json.loads(open(report_name, 'r').read()[:-1] + ']')


overall_results = {}

for vendor_name in os.listdir(report_folder):
    baselines_results = glob.glob(f'{report_folder}/{vendor_name}/baselines_*')
    device_count = len(baselines_results)

    vendor_score = 0
    gpu_count = 0

    for idx, device_reports in enumerate(baselines_results):
        reports = read_results(device_reports)

        device_score = 0
        bench_count = 0

        for bench_result in reports:
            device_score += score(bench_result)
            bench_count += 1

        vendor_score += device_score / bench_count
        gpu_count += 1

    overall_results[vendor_name] = gpu_count * 100 / vendor_score


print(json.dumps(overall_results, indent=4))



# add_timestep	algo	all.avg	all.count	all.max	all.min	all.sd	all.unit
# alpha	arch	batch_loss	batch_size	beam_size	beta1	bptt	bucketing
# checkpoint	checkpoint_interval	checkpoint_model_dir	clip	clip_param
# content_weight	cov_penalty_factor	cpu_cores	cuda	cuda_deterministic
# cudnn	data	dataroot	dataset	dataset_dir	devices	disable_eval
# dist_url	dropout	dtype	dynamic_loss_scale	emsize	entropy_coef
# env_name	epoch_loss	epochs	eps
# eval.avg	eval.count	eval.max	eval.min	eval.sd	eval.unit
# eval_batch_size	eval_interval	evaluation	factors	fp16
# gamma	gpu	grad_clip	hostname	imageSize	image_size	index
# iteration	jr_id	keep_checkpoints	layers	learning_rate	len_norm_const
# len_norm_factor
# loading_data.avg	loading_data.count	loading_data.max	loading_data.min	loading_data.sd	loading_data.unit
# local_rank	log_dir	log_interval	lr	math	max_grad_norm	max_length_train_item	max_length_val
# max_size
# metrics.errD_real	metrics.errG	metrics.errG_fake	metrics.psnr	metrics.val_loss	metrics.value_loss
# min_length_train_item	min_length_val	model	model_config	name	ndf	negative_samples	netD	netG	ngf
# ngpu	nhid	nlayers	no_checks	no_save	num_env_steps	num_mini_batch	num_processes	num_steps	number
# nz	onnx_export	opt_level	optimization_config	outf	port	ppo_epoch	print_freq	processes	prof
# rank	recurrent_policy	render	repeat	results_dir	resume	save	save_all	save_dir	save_freq
# save_interval	save_model_dir	smoothing	start_epoch	static_loss_scale	style_image	style_size	style_weight
# subcommand	target_bleu	tau	testBatchSize	threshold	tied	topk
# train_item.avg	train_item.count	train_item.max	train_item.min	train_item.sd	train_item.unit
# train_item_item.avg	train_item_item.max	train_item_item.min	train_item_item.range	train_item_item.unit
# unique_id	upscale_factor	use_gae	use_linear_clip_decay	use_linear_lr_decay	value_loss_coef
# vcd	version	vis	vis_interval	workers	world_size


#df = json_normalize(reports)
#df.to_csv('all_aggregated.csv')
selected_columns = {
    'gpu', 'hostname', 'index', 'name'
    ,'train_item.avg'
    ,'train_item.count'
    ,'train_item.max'
    ,'train_item.min'
    ,'train_item.sd'
    ,'train_item.unit'
    ,'train_item_item.avg'
    ,'train_item_item.max'
    ,'train_item_item.min'
    ,'train_item_item.range'
    ,'train_item_item.unit'
    ,'unique_id'
    ,'version'
    ,'workers'
    ,'world_size'
    ,'vcd'
    ,'devices'
    ,'cpu_cores'
    ,'batch_size'
    ,'all.avg'
    ,'all.count'
    ,'all.max'
    ,'all.min'
    ,'all.sd'
    ,'eval.avg'
    ,'eval.count'
    ,'eval.max'
    ,'eval.min'
    ,'eval.sd'
    ,'fp16'
    ,'loading_data.avg'
    ,'loading_data.count'
    ,'loading_data.max'
    ,'loading_data.min'
    ,'loading_data.sd'
}

#perf_report = df[selected_columns]
#perf_report.to_csv('perf_report.csv', index=False)


