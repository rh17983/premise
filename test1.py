import os

exp_dir = "/data/1427395274-10.40.7.173-networkloss_sprout_240"

exp_name = os.path.split(exp_dir)[-1]
exp_id = int(exp_name.split('-')[0])

print(exp_name)
print(exp_id)
