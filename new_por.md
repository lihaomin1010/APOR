# 代码

https://github.com/lihaomin1010/APOR

改的地方在new_main_por.py

# 论文
iql https://arxiv.org/abs/2110.06169

por https://arxiv.org/abs/2210.08323

# 实验
d4rl https://github.com/Farama-Foundation/d4rl/wiki/Tasks GYM部分

lhm：medium-expert

xyr：expert

zrh：medium-replay


## 命令
### por-q
--type por_q的时候 这里有个预训练过程，我没搞清，我再看看，大家一起看下 参数里面是pretrain

python main_por.py --env_name hopper-medium-expert-v2 --type por_q --tau 0.7 --alpha 3.0 --eval_period 5000 --n_eval_episodes 10 --policy_lr 0.001 --layer_norm --seed 11 --pretrain

### por-r
python main_por.py --env_name hopper-medium-expert-v2 --type por_r --tau 0.7 --alpha 3.0 --eval_period 5000 --n_eval_episodes 10 --policy_lr 0.001 --layer_norm --seed 11

### my
python new_main_por.py --env_name hopper-medium-expert-v2 --tau 0.7 --alpha 3.0 --eval_period 5000 --n_eval_episodes 10 --policy_lr 0.001 --layer_norm --seed 11

这部分的tau和alpha参数在不同的环境里有不同的参考值，在por论文附录里面有


# 任务
lhm：看论文idea撞车没 

xyr：看por的消融实验和其他实验设计思路 

zrh：看方差大小，重点看alpha这个权重参数

# 思考
1.是否有更多的trick，例如hiql中的压缩映射
2.一个完备的数学推导
