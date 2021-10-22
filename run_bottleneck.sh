#!/bin/bash
seed=$1
num_rooms=$2
task_evol=$3
env="NBottleneckClass-v0"

echo "===========================Running baselines==========================="

echo "Running Online Q learning | "
python tabular_expts.py --agent "qlearning" --num-rooms $num_rooms --gamma 0.99 --lifetime 1000000 --env-name $env --length 5 --task-evolution $task_evol --epsilon 0.35 --lr 0.5 --seed $seed

echo "Running Dyna Q learning | $env"
python tabular_expts.py --agent "dynalearning" --num-rooms $num_rooms --gamma 0.99 --lifetime 1000000 --env-name $env --length 5 --task-evolution $task_evol --epsilon 0.25 --lr 0.35  --steps 5 --seed $seed

echo "Running Q learning w/ replay | $env"
python tabular_expts.py --agent "replaylearning" --num-rooms $num_rooms --gamma 0.99 --lifetime 1000000 --env-name $env --length 5 --task-evolution $task_evol --epsilon 0.2 --lr 0.3 --steps 5 --seed $seed

echo "Running N-step TD | $env"
python tabular_expts.py --agent "nstepTDlearning" --num-rooms $num_rooms --gamma 0.99 --lifetime 1000000 --env-name $env --length 5 --task-evolution $task_evol --epsilon 0.2 --lr 0.2 --steps 5 --seed $seed

echo "Running Vanilla Policy gradient | $env"
python tabular_expts.py --agent "policygradient" --num-rooms $num_rooms --gamma 0.99 --lifetime 1000000 --env-name $env --length 5 --task-evolution $task_evol --steps_pi 5 --steps 5 --epsilon 0.1 --lr 0.01 --lr_pi 0.01 --seed $seed

echo "===========================Running our proposed models==========================="
echo "Running On-Policy Rho | "
python tabular_expts.py --agent "onpolicyrho" --num-rooms $num_rooms --lifetime 1000000  --env-name $env --length 5 --task-evolution $task_evol --steps 10 --epsilon 0.1 --lr 0.3  --seed $seed --solver "numpy_solver"

echo "Running Off-policy Rho | $env"
python tabular_expts.py --agent "offpolicyrho" --num-rooms $num_rooms --lifetime 1000000 --env-name $env --length 5 --task-evolution $task_evol --steps 5 --epsilon 0.1 --lr 0.35 --seed $seed --solver "numpy_solver"

echo "Running rho-policy gradient | $env"
python tabular_expts.py --agent "rhogradient" --num-rooms $num_rooms --lifetime 1000000 --env-name $env --length 5 --task-evolution $task_evol --steps 10 --epsilon 0.15 --lr 0.5 --seed $seed --solver "numpy_solver"

echo "===========================$env experiments done==========================="