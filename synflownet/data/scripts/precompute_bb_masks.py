import argparse
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from rdkit import Chem

from synflownet.utils.synthesis_utils import Reaction


def get_precomputed_rxn_bb_masks(bb_path: Path, template_path: Path):
    bb_file_root_path = bb_path.parent.absolute()
    precomputed_bb_masks_path = bb_file_root_path / f"masks__{bb_path.stem}__X__{template_path.stem}.pkl"
    if not precomputed_bb_masks_path.exists():
        precomputed_bb_masks = compute_bb_masks(
            bb_path=bb_path,
            template_path=template_path,
            output_path=precomputed_bb_masks_path,
            add_Hs=template_path.stem == "real",
        )
    with open(precomputed_bb_masks_path, "rb") as f:
        precomputed_bb_masks = pickle.load(f)

    return precomputed_bb_masks


def compute_bb_masks(bb_path, template_path, output_path, add_Hs: bool = False):
    with open(bb_path) as f:
        building_blocks = f.readlines()

    with open(template_path) as f:
        reaction_templates = f.read().splitlines()

    reactions = [Reaction(template=t, building_blocks=building_blocks) for t in reaction_templates]
    bimolecular_reactions = [r for r in reactions if r.num_reactants == 2]
    building_blocks_mols = [Chem.MolFromSmiles(bb) for bb in building_blocks]
    if add_Hs:
        building_blocks_mols = [Chem.AddHs(mol) for mol in building_blocks_mols]

    logger.info(f"Computing masks with:\ntemplate_path: {template_path}\nbb_path: {bb_path}")
    masks = np.zeros((2, len(bimolecular_reactions), len(building_blocks)))
    for rxn_i in range(len(bimolecular_reactions)):
        reaction = bimolecular_reactions[rxn_i]
        reactants = reaction.rxn.GetReactants()
        for bb_j, bb in enumerate(building_blocks_mols):
            if bb is None:
                logger.info(f"Invalid: bb_j: {bb_j}, building_blocks[bb_j]: {building_blocks[bb_j]}")
            if bb.HasSubstructMatch(reactants[0]):
                masks[0, rxn_i, bb_j] = 1
            if bb.HasSubstructMatch(reactants[1]):
                masks[1, rxn_i, bb_j] = 1
        logger.info(
            f"{rxn_i} of {len(bimolecular_reactions)} -- "
            f"Reactant1: {masks[0, rxn_i, :].sum() / masks.shape[2] * 100.0:.2f}% match -- "
            f"Reactant2: {masks[1, rxn_i, :].sum() / masks.shape[2] * 100.0:.2f}% match"
        )

    logger.info(f"Saving precomputed masks of shape={masks.shape} to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(masks, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        type=str,
        default=Path(__file__).parent / "precomputed_enamine_bb_270k_HB_matching_short_0.1_sanitized.pkl",
    )
    args = parser.parse_args()

    compute_bb_masks(args.out_path)

    print("Done!")
    with open(args.out_path, "rb") as f:
        masks = pickle.load(f)
    print(masks.shape)
