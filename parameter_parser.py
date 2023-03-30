import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ##################################
    parser.add_argument('--cuda', type=int, default=0, help='specify gpu')
    parser.add_argument('--exp', type=str, default='class_mem_infer_siamese',
                        choices=['class_mem_infer_siamese', 'class_mem_infer_meta'])
    parser.add_argument('--target_model', type=str, default='siamesenet', choices=['protonet', 'relationnet', 'siamesenet'])
    parser.add_argument('--shadow_model', type=str, default='siamesenet', choices=['protonet', 'relationnet', 'siamesenet'])

    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--feature_extractor', type=str, default='SCNN', 
                        choices=['SCNN', 'MobileNet', 'GoogleNet', 'ResNet18', 'ResNet50'])
    parser.add_argument('--attack_model', type=str, default='MLP', choices=['MLP', 'RF', 'AUC'])

   ######################### dataset related parameters ##################################
    parser.add_argument('--dataset_name', type=str, default='vggface2',
                        choices=['vggface2', 'webface', 'umdfaces', 'celeba'])
    parser.add_argument('--shadow_dataset_name', type=str, default='vggface2',
                        choices=['vggface2', 'webface', 'umdfaces', 'celeba'])
    parser.add_argument('--image_size', type=int, default=32, choices=[32, 64, 96, 112])
    parser.add_argument('--dataset_task', type=int, default=5, help=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--num_of_samples', type=int, default=100, choices=[20, 32, 64, 96, 100])

    ########################## probe controlling parameters ##############################
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--is_split', type=str2bool, default=False)
    parser.add_argument('--is_save_results', type=str2bool, default=True)
    parser.add_argument('--is_train_target', type=str2bool, default=False)
    parser.add_argument('--is_train_shadow', type=str2bool, default=False)
    parser.add_argument('--is_disjoint_train', type=str2bool, default=True)
    parser.add_argument('--is_balance_attack', type=str2bool, default=True)
    parser.add_argument('--is_generate_probe', type=str2bool, default=True)

    parser.add_argument('--lpips_backbone', type=str, default='vgg', choices=['alex', 'squeeze', 'vgg'])
    parser.add_argument('--is_normalize_similarity', type=str2bool, default=False)
    parser.add_argument('--is_similarity_aided', type=str2bool, default=True)
    parser.add_argument('--is_use_image_similarity', type=str2bool, default=True)
    parser.add_argument('--image_similarity_name', type=str, default="cosine",
                        choices=['lpips', 'cosine', 'mse', 'msssim', 'none', 'ssim'])
    parser.add_argument('--image_similarity_level', type=int, default=5,
                        choices=[0, 1, 2, 3, 4, 5], help="lower value means more similar images")

    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--is_known_embedding', type=str2bool, default=False,
                        help="Whether the adversary can access the embeddings.")
    parser.add_argument('--embedding_metric', type=str, default="Cu", choices=['Cu', 'Pu', 'CuPu'])
    ########################## target siamese parameters ############################
    parser.add_argument('--num_iter', type=int, default=1000, help='vary number of iteration for different dataset. 1000. 50000')
    parser.add_argument('--test_times', type=int, default=400)
    parser.add_argument('--probe_times', type=int, default=400)
    parser.add_argument('--num_training_shots', type=int, default=300, help="Number of samples per training class")
    parser.add_argument('--batch_size', type=int, default=64)

    ########################## target meta parameters ###############################
    parser.add_argument('--train_num_epochs', type=int, default=100)
    parser.add_argument('--train_num_query', type=int, default=5)
    parser.add_argument('--train_num_task', type=int, default=100)
    parser.add_argument('--test_num_query', type=int, default=5)
    parser.add_argument('--test_num_task', type=int, default=80)

    ########################## query/probe parameters ###############################
    parser.add_argument('--is_sort_query', type=str2bool, default=True)
    parser.add_argument('--is_disjoint_probe', type=str2bool, default=True)
    parser.add_argument('--probe_ways', type=int, default=5, help='number of ways in probe support set')
    parser.add_argument('--probe_shot', type=int, default=5, help='number of shot in probe support set')
    parser.add_argument('--probe_num_query', type=int, default=5, help='number of query samples in probe query set')
    parser.add_argument('--probe_num_task', type=int, default=100, help='number of tasks in probe set')

    ########################## dp (training) defense parameters ##############################
    parser.add_argument('--is_dp_defense', type=str2bool, default=False)
    parser.add_argument("--sigma", type=float, default=0.5, metavar="S", help="Noise multiplier (default 1.0)", )
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0,
                        metavar="C", help="Clip per-sample gradients to this norm (default 1.0)", )
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta (default: 1e-5)", )
    parser.add_argument("-sr", "--sample-rate", type=float, default=0.001,
                        metavar="SR", help="sample rate used for batch construction (default: 0.001)", )
    parser.add_argument("--secure_rng", action="store_true", default=False,
                        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost", )
    parser.add_argument("--lr", type=float, default=0.005, metavar="LR", help="learning rate (default: .1)", )
    parser.add_argument('--n_accumulation_steps', type=float, default=1)

    ########################## adv (input) defense parameters ##############################
    parser.add_argument('--is_adv_defense', type=str2bool, default=False)
    parser.add_argument('--fawkes_mode', type=str, default="none", help=['none', 'low', 'mid', 'high'])

    ########################## noise (output) defense parameters ##############################
    parser.add_argument('--is_noise_defense', type=str2bool, default=True)
    parser.add_argument('--noise_std', type=float, default=0.8)

    ########################## memguard (adaptive) defense parameters ##############################
    parser.add_argument('--is_memguard_defense', type=str2bool, default=False,
                        help='Add perturbations on the similarity scores to evade face-auditor.')
    return vars(parser.parse_args())
