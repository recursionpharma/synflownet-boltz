import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol

from synflownet.data.async_sql_databases import PersistentReplayBuffer, RewardQueue


def simple_reward_fn(mols: list[RDMol]) -> list[float]:
    """Count number of nitrogen and oxygen atoms in each molecule"""
    return [sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ["N", "O"]) / 10.0 for mol in mols]


def process_batch(
    reward_queue: RewardQueue,
    persistent_buffer: PersistentReplayBuffer,
    batch_size: int = 64,
):
    """Process one batch of molecules from queue"""
    try:
        print(f"Current db sizes -- reward queue: {reward_queue.get_db_size():,}, persistent buffer: {persistent_buffer.get_db_size():,}")
        # Pop batch from queue
        entries = reward_queue.pop(batch_size)
        if len(entries) == 0:
            raise ValueError("No samples found in reward queue.")
        ids, smiles, trajs, none_rewards, timestamps, infos = entries

        # Calculate rewards
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        rewards = simple_reward_fn(mols)

        # Push to persistent buffer
        persistent_buffer.push(smiles, trajs, rewards, infos)

        print(f"Processed {len(smiles)} trajectories.")
        print(f"New db sizes -- reward queue: {reward_queue.get_db_size():,}, persistent buffer: {persistent_buffer.get_db_size():,}")

    except ValueError as e:
        print(f"Error processing batch: {e}")
        return False

    return True


def main(db_root_path: Path):
    # Initialize databases
    reward_queue = RewardQueue(db_path=db_root_path / "reward_queue.db", max_size=400)
    persistent_buffer = PersistentReplayBuffer(db_path=db_root_path / "persistent_replay.db")

    print(f"Reward queue size: {reward_queue.get_db_size()}")
    print(f"Persistent buffer size: {persistent_buffer.get_db_size()}")
    while True:
        # user_input = input("\nReady to label, enter 'y' to label a batch or 'n' to terminate: ")
        # if user_input.lower() == 'n':
        #     break
        # elif user_input.lower() == 'y':
        success = process_batch(reward_queue, persistent_buffer)
        if not success:
            print("Waiting 5 seconds...")
            time.sleep(5)
        # else:
        #     print("Invalid input, please enter 'y' or 'n'")

    print("Closing.")


if __name__ == "__main__":
    main(Path("./tmp_dbs"))
