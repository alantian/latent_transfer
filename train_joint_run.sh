#!/usr/bin/env bash

#######################################
# Bash3 Boilerplate Start
# copied from https://kvz.io/blog/2013/11/21/bash-best-practices/

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

arg1="${1:-}"
# Bash3 Boilerplate End
#######################################

function mnist2mnist_exp1 () {
    n_latent_shared=$1
    pairing_number=$2   # -1 for everything
    prior_loss_align_beta=$3   # 0.02 is good value
    mean_recons_A_align_beta=$4  # these two values: 0.5
    mean_recons_B_align_beta=$5
    mean_recons_A_to_B_align_beta=$6  # these two vlaues: good is 0.1
    mean_recons_B_to_A_align_beta=$7
    mean_recons_A_to_B_align_free_budget=$8  # hard to say. 0.0 is good start.
    mean_recons_B_to_A_align_free_budget=$9

    if [ "$#" -ne 9 ]; then
      echo "Illegal number of parameters"
    fi

    run_ml_docker python3 ./train_joint.py  --n_iters_per_eval 1000 \
        --default_scratch "~/workspace/scratch/latent_transfer/" \
        --config joint_exp_mnist_family_parameterized \
        --exp_uid "_mnist2mnist_exp1_nl$1_pn$2_b_$3_$4_$5_$6_$7_fb_$8_$9" \
        --mnist_family_config_A mnist_0_nlatent100_xsigma1 \
        --mnist_family_config_B mnist_0_nlatent100_xsigma1 \
        --mnist_family_config_classifier_A mnist_classifier_0 \
        --mnist_family_config_classifier_B mnist_classifier_0 \
        --n_latent 100 \
        --shuffle_only_once_for_paired_data=true \
        --n_latent_shared $n_latent_shared \
        --pairing_number $pairing_number \
        --prior_loss_align_beta $prior_loss_align_beta \
        --mean_recons_A_align_beta $mean_recons_A_align_beta \
        --mean_recons_B_align_beta $mean_recons_B_align_beta \
        --mean_recons_A_to_B_align_beta $mean_recons_A_to_B_align_beta \
        --mean_recons_B_to_A_align_beta $mean_recons_B_to_A_align_beta \
        --mean_recons_A_to_B_align_free_budget $mean_recons_A_to_B_align_free_budget \
        --mean_recons_B_to_A_align_free_budget $mean_recons_B_to_A_align_free_budget \
        ;
}

func_name=$1
shift 1
$func_name "$@"