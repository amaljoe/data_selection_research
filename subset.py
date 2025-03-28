import submodlib as sb
import pickle
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "subset")

def create_subset(data_sijs, utility_name, k=0.3):
    subset_name = f'{utility_name}_{k}'
    cache_file = os.path.join(cache_dir, f"{subset_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f'Subset: {subset_name} found in cache, loading from cache ✅')
        with open(cache_file, 'rb') as f:
            return pickle.load(f), subset_name
    print(f'Subset: {subset_name} not found in cache, computing now 🏃')
    # scale matrix
    data_sijs = np.random.randn(len(data_sijs), len(data_sijs))
    data_sijs = (data_sijs - data_sijs.min()) / (data_sijs.max() - data_sijs.min())
    n, _ = data_sijs.shape
    # use facility location to find subset
    fl = sb.functions.facilityLocation.FacilityLocationFunction(n, mode='dense', sijs=data_sijs, separate_rep=False)
    subset = fl.maximize(budget=int(k * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, show_progress=True)

    with open(cache_file, 'wb') as f:
        pickle.dump(subset, f)
    print(f'Subset: {subset_name} computed and saved to cache ✅')
    return subset, subset_name

def get_subset(subset, prompts, references):
    subset_indices = np.array(subset)[:, 0].astype(int)
    subset_prompts = [prompts[i] for i in subset_indices]
    subset_references = [references[i] for i in subset_indices]
    return subset_prompts, subset_references

if __name__ == '__main__':
    from data_loader import get_mix_instruct
    from utility_functions.delift_se import get_delift_se_utility

    prompts, references, ds_name = get_mix_instruct("train", 1000)
    utility, utility_name = get_delift_se_utility(prompts, references, ds_name)
    subset, subset_name = create_subset(utility, utility_name)
    s_prompts, s_references = get_subset(subset, prompts, references)
