import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess  # nosec B404
import sys
import time
import traceback
from pathlib import Path

import medchem as mc
import pandas as pd
import yaml
from loguru import logger
from rdkit import Chem


def generate_unique_worker_id():
    timestamp_ms = int(time.time() * 1000) % 10_000_000
    array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) % 100
    rand = random.randint(0, 99)
    id_str = f"{array_task_id:02d}_{timestamp_ms:07d}_{rand:02d}"
    return id_str


def main(config: dict):
    worker_id = generate_unique_worker_id()

    # Setting up logging for this module only
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")
    logger.info(f"Boltz2 reward worker started with worker ID: {worker_id}")

    # Print config
    logger.info(f"Executing Boltz2 reward worker with config:\n{yaml.dump(config)}")

    # Initialise reward queue, replay buffer, and reward cache
    reward_queue, replay_buffer, reward_cache = initialise_dbs(config)

    #
    # Retrieve protein sequence and msa file path
    #
    # To ensure the anonymity of the targets, we keep the target to data mapping in a separate YAML file.
    # If the path to this file is provided in the config, we will use it otherwise we will use the default path.
    #
    if "path_to_target_to_data" in config:
        logger.info(f"Using custom path to target_to_data.yaml: {config['path_to_target_to_data']}")
        target_to_data_path = Path(config["path_to_target_to_data"])
    else:
        logger.info("Using default path to target_to_data.yaml: data/target_to_data.yaml")
        target_to_data_path = Path("data/target_to_data.yaml")

    with open(target_to_data_path) as f:
        target_to_data = yaml.safe_load(f)

    protein_sequences = []
    msa_file_paths = []
    for target in config["targets"]:
        protein_sequences.append(target_to_data[target]["protein_sequence"])
        msa_file_paths.append(target_to_data[target]["msa_file_path"])

    assert len(protein_sequences) == len(msa_file_paths), "Every protein sequence must have a corresponding MSA file path"
    for protein_seq, msa_path in zip(protein_sequences, msa_file_paths):
        logger.info(f"Protein sequence: {protein_seq}, MSA file path: {msa_path}")

    current_batch = 0
    while True:
        # Removes next batch from reward queue
        try:
            batch = reward_queue.pop(batch_size=config["worker_batch_size"])
            if len(batch) == 0:
                raise Exception("No batches to process")
        except Exception:
            logger.error(f"Error popping from reward queue:\n{traceback.format_exc()}")
            time.sleep(5)  # Back off before retrying
            continue  # nosec B112

        batch_df = pd.DataFrame(
            {
                "id": batch[0],
                "SMILES": batch[1],
                "traj": batch[2],
                "reward": batch[3],
                "timestamp": batch[4],
            }
        )
        if batch_df.empty:
            logger.info("No batches to process")
            time.sleep(5)
            continue  # nosec B112

        # Retrieve cached results based on smiles
        smiles_list = batch_df["SMILES"].dropna().unique().tolist()
        cached_entries_df, uncached_entries_df = get_cache_hits(reward_cache, batch_df, smiles_list)
        logger.info(
            f"Processing batch {current_batch} of size {len(batch_df)}:"
            f" Number of unique smiles: {len(batch_df['SMILES'].unique()):,},"
            f" Cached entries: {cached_entries_df.shape[0]:,},"
            f" Uncached entries: {uncached_entries_df.shape[0]:,}"
        )

        # For cached entries (if not empty), push retrieved data to the replay buffer
        if not cached_entries_df.empty:
            if cached_entries_df["reward"].isna().any():
                logger.error(f"Cached entries have NaN rewards: {len(cached_entries_df[cached_entries_df['reward'].isna()])}")
                continue  # nosec B112

            smiles = cached_entries_df["SMILES"].tolist()
            trajs = cached_entries_df["traj"].tolist()
            rewards = cached_entries_df["reward"].tolist()
            infos = cached_entries_df["info"].tolist()

            replay_buffer.push(smiles, trajs, rewards, infos)

        # For uncached entries (if not empty), compute the rewards
        if not uncached_entries_df.empty:
            uncached_entries_len = len(uncached_entries_df)
            try:
                #
                # We will need to adjust this part if we want to use multiple targets
                #
                uncached_rewards_df = compute_rewards_with_boltz(
                    df=uncached_entries_df,
                    protein_sequence=protein_sequences,
                    msa_file_path=msa_file_paths,
                    worker_id=worker_id,
                )
            except Exception:
                logger.error(f"Error computing rewards with Boltz2:\n{traceback.format_exc()}")
                continue  # nosec B112

            if len(uncached_rewards_df) != uncached_entries_len:
                logger.error(
                    f"Something went wrong: size of uncached_rewards_df "
                    f"has changed from {uncached_entries_len} -> {len(uncached_rewards_df)}"
                )
                continue  # nosec B112

            if uncached_rewards_df["reward"].isna().any():
                logger.error(f"Uncached entries have NaN rewards: {len(uncached_entries_df[uncached_entries_df['reward'].isna()])}")
                continue  # nosec B112

            # Cache the results using RewardCache
            uncached_rewards_to_cache_df = uncached_rewards_df[["SMILES", "reward", "info"]]
            uncached_rewards_to_cache_df = uncached_rewards_to_cache_df.dropna(subset=["SMILES", "reward"])
            uncached_rewards_to_cache_df = uncached_rewards_to_cache_df.drop_duplicates(subset=["SMILES"])

            try:
                # Convert DataFrame to list of tuples for RewardCache.insert_entries
                cache_entries = [
                    (row["SMILES"], row["reward"], row["info"] if pd.notna(row["info"]) else "")
                    for _, row in uncached_rewards_to_cache_df.iterrows()
                ]
                reward_cache.insert_entries(cache_entries)
            except Exception:
                logger.error(f"Error caching new rewards:\n{traceback.format_exc()}")

            # Store uncached rewarded trajs
            smiles = uncached_rewards_df["SMILES"].tolist()
            trajs = uncached_rewards_df["traj"].tolist()
            rewards = uncached_rewards_df["reward"].tolist()
            infos = uncached_rewards_df["info"].tolist()
            try:
                replay_buffer.push(smiles, trajs, rewards, infos)
            except Exception:
                logger.error(f"Error pushing to replay buffer:\n{traceback.format_exc()}")

        # increment the current batch
        logger.info(f"Batch {current_batch} completed")
        current_batch += 1


def initialise_dbs(config: dict):
    # Direct import to avoid having to install synflownet to run this script
    spec = importlib.util.spec_from_file_location(
        name="async_sql_databases",
        location=Path(__file__).absolute().parents[2] / "synflownet/data/async_sql_databases.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Instantiate RewardQueue, PersistentReplayBuffer, and RewardCache
    reward_queue = module.RewardQueue(
        db_path=config["reward_queue_path"],
        max_size=config["reward_queue_max_size"],
        max_batch_size=config["worker_batch_size"],
    )
    replay_buffer = module.PersistentReplayBuffer(
        db_path=config["persistent_replay_path"],
        max_size=config["persistent_replay_max_size"],
        max_batch_size=config["worker_batch_size"],
    )
    reward_cache = module.RewardCache(db_path=config["reward_cache_path"])

    return reward_queue, replay_buffer, reward_cache


def get_cache_hits(reward_cache, batch_df: pd.DataFrame, smiles_list: list[str]):
    try:
        # Use RewardCache to get hits
        cache_results = reward_cache.get_hits(smiles_list)
    except Exception:
        logger.error(f"Database error occurred:\n{traceback.format_exc()}")
        raise

    # Create dictionaries for O(1) lookup
    cached_smiles = [row[0] for row in cache_results]
    smiles_to_reward = {row[0]: row[1] for row in cache_results}
    smiles_to_info = {row[0]: row[2] for row in cache_results}

    # Add rewards and info to batch_df where smiles matches
    batch_df["is_cached"] = batch_df["SMILES"].isin(cached_smiles)
    batch_df["reward"] = batch_df["SMILES"].map(lambda x: smiles_to_reward.get(x, None))
    batch_df["info"] = batch_df["SMILES"].map(lambda x: smiles_to_info.get(x, ""))

    # Split the batch into cached and uncached entries, keeping only unique SMILES
    cached_entries_df = batch_df[batch_df["is_cached"] == True].drop_duplicates(subset=["SMILES"]).reset_index(drop=True)  # noqa: E712
    uncached_entries_df = batch_df[batch_df["is_cached"] == False].drop_duplicates(subset=["SMILES"]).reset_index(drop=True)  # noqa: E712

    return cached_entries_df, uncached_entries_df


def prepare_dirs(input_dir: str, output_dir: str) -> None:
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    if Path(input_dir).exists():
        shutil.rmtree(input_dir)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


def prepare_boltz2_yaml_input_file(protein_sequence: str | list[str], ligand_smiles: str, yaml_input_path: str, msa_file_path: str | list[str]) -> None:
    if not isinstance(protein_sequence, list):
        protein_sequence = [protein_sequence]
    if not isinstance(msa_file_path, list):
        msa_file_path = [msa_file_path]
    assert len(protein_sequence) == len(msa_file_path), "Protein sequence and MSA file paths must have the same length"

    yaml_dict = {"sequences": [], "properties": []}

    unicode_string_ordinal_id = 65

    for protein_seq, msa_path in zip(protein_sequence, msa_file_path):
        protein_id = chr(unicode_string_ordinal_id)
        unicode_string_ordinal_id += 1
        yaml_dict["sequences"].append(
            {"protein": {"id": protein_id, "sequence": protein_seq, "msa": msa_path, "cyclic": False}}
        )

    ligand_id = chr(unicode_string_ordinal_id)
    yaml_dict["sequences"].append({"ligand": {"id": ligand_id, "smiles": ligand_smiles}})
    yaml_dict["properties"].append({"affinity": {"binder": ligand_id}})

#    yaml_dict = {
#        "sequences": [
#            {"protein": {"id": "A", "sequence": protein_sequence, "msa": msa_file_path, "cyclic": False}},
#            {"ligand": {"id": "B", "smiles": ligand_smiles}},
#        ],
#        "properties": [{"affinity": {"binder": "B"}}],
#    }

    with open(yaml_input_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)


def run_boltz_inference(input_dir: str, output_dir: str, cache_dir: str = "~/project/boltz_cache", verbose: bool = False):
    # run cli command
    command_list = ["boltz", "predict", input_dir, "--out_dir", output_dir, "--cache", cache_dir]

    # handle verbosity
    try:
        if verbose:
            subprocess.run(command_list, check=True)
        else:
            subprocess.run(command_list, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Boltz command failed with error:\n{e.stderr}")
        raise


def parse_info(info_str: str) -> dict:
    """
    Parses the JSON-encoded `info` string into a dictionary with correctly typed values.
    """
    parsed_dict = {}
    if info_str is None or info_str == "":
        return {"status": "FAILED"}

    try:
        raw_dict = json.loads(info_str)
        parsed_dict["status"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Error parsing info: {e}")
        return {"status": "FAILED"}

    for key, type_val in raw_dict.items():
        type_str, _, val_str = type_val.partition(":")
        if type_str == "int":
            parsed_val = int(val_str)
        elif type_str == "float":
            parsed_val = float(val_str)
        elif type_str == "bool":
            parsed_val = val_str.lower() == "true"
        elif type_str == "NoneType":
            parsed_val = None
        else:  # fallback to string
            parsed_val = val_str
        parsed_dict[key] = parsed_val

    return parsed_dict


def compute_reward(aff: float, prob: float, smiles: str) -> float:
    """
    Computes the reward for a given affinity, probability, and SMILES.
    """
    normalized_aff = max(0.0, (aff * -1 + 2.0) / 4.0)
    lily_mask = mc.functional.lilly_demerit_filter(mols=[Chem.MolFromSmiles(smiles)], n_jobs=-1, progress=False, return_idx=False)
    assert len(lily_mask.shape) == 1, "Lilly mask should be a 1D array"
    assert lily_mask.shape[0] == 1, "Lilly mask should have only one element"
    lily_mask = float(lily_mask[0])
    assert lily_mask == 0.0 or lily_mask == 1.0, "Lilly mask should be 0.0 or 1.0"

    return float(normalized_aff * prob * lily_mask)


def collect_boltz_results(input_dir: str, predictions_path: str, query_name: str) -> dict:
    """
    Collects and processes Boltz2 inference results from the predictions directory.
    """
    result_dict = {}
    info_dict = {}

    # Gets SMILES processed by Boltz2 from input_file
    input_file_path = Path(input_dir) / f"{query_name}.yaml"
    with open(input_file_path) as f:
        input_file_content = yaml.safe_load(f)
    smiles = input_file_content["sequences"][1]["ligand"]["smiles"]
    result_dict["SMILES"] = smiles

    # Get both affinity and confidence files
    query_path = Path(predictions_path) / f"{query_name}"

    affinity_file_paths = list(query_path.rglob("affinity_*.json"))
    assert len(affinity_file_paths) == 1, f"Expected exactly one affinity file, got {len(affinity_file_paths)}"
    affinity_file_path = affinity_file_paths[0]

    confidence_file_paths = list(query_path.rglob("confidence_*.json"))
    assert len(confidence_file_paths) == 1, f"Expected exactly one confidence file, got {len(confidence_file_paths)}"
    confidence_file_path = confidence_file_paths[0]

    with open(affinity_file_path) as f:
        affinity_pred = json.load(f)
    with open(confidence_file_path) as f:
        confidence_pred = json.load(f)

    # Collect info and compute reward
    info_dict.update(affinity_pred)
    info_dict["confidence_score"] = confidence_pred["confidence_score"]
    result_dict["info"] = json.dumps({k: f"{type(info_dict[k]).__name__}:{info_dict[k]}" for k in info_dict}, default=str)

    result_dict["reward"] = compute_reward(
        float(affinity_pred["affinity_pred_value1"]), float(affinity_pred["affinity_probability_binary1"]), smiles
    )

    return result_dict


def compute_rewards_with_boltz(df: pd.DataFrame, protein_sequence: str | list[str], msa_file_path: str | list[str], worker_id: int) -> pd.DataFrame:
    assert "SMILES" in df.columns, "SMILES column is required"

    # Create temporary directory for worker
    tmp_dir = Path(f"./reward_workers_tmp_dirs/tmp_worker_{worker_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Run Boltz2 to get rewards
    df["reward"] = 0.0
    df["info"] = ""
    df["query_name"] = None
    input_dir = tmp_dir / "input_files"
    output_dir = tmp_dir / "output_files"
    logger.info(f"Running Boltz2 inference for {len(df)} molecules")

    # Prepare input files for Boltz-2 inference pipeline
    prepare_dirs(input_dir, output_dir)
    for i in df.index:
        query_name = f"query{i}"
        yaml_input_path = f"{input_dir}/{query_name}.yaml"
        prepare_boltz2_yaml_input_file(protein_sequence, df.at[i, "SMILES"], yaml_input_path, msa_file_path)
        df.at[i, "query_name"] = query_name

    # Run inference
    run_boltz_inference(input_dir=input_dir, output_dir=output_dir, verbose=True)

    # Collect results with rewards and info
    run_name = Path(input_dir).stem
    predictions_path = f"{output_dir}/boltz_results_{run_name}/predictions"
    for i in df.index:
        try:
            result_dict = collect_boltz_results(input_dir, predictions_path, df.at[i, "query_name"])
            if result_dict["SMILES"] == df.at[i, "SMILES"]:
                df.at[i, "reward"] = result_dict["reward"]
                df.at[i, "info"] = result_dict["info"]
            else:
                logger.warning(f"SMILES mismatch for query {i}")
        except Exception:
            logger.warning(
                f"Result could not be collected for smiles number {i}:" f"\nSMILES: {df.at[i, 'SMILES']}" f"\n{traceback.format_exc()}"
            )

    # Clean up temporary directory
    shutil.rmtree(tmp_dir)

    return df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Boltz2 reward worker")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)
