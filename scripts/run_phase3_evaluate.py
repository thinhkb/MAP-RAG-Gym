from collections import Counter

from map_rag_gym.utils.io import read_json

runs = read_json("outputs/phase1_demo.json")
mods = Counter()
for run in runs:
    for step in run["steps"]:
        mods[step["module"]] += 1
print("Module counts:", dict(mods))
for run in runs:
    print(run["workflow_id"], run["final_scores"])
