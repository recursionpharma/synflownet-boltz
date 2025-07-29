from rdkit import Chem
with Chem.SDMolSupplier('/home/ubuntu/synflownet/src/synflownet/data/building_blocks/Pharmaron-catalogue for aromatic acids.sdf') as suppl:
    # Save the SMILES of the molecules in the SDF file to a text file
    # This is useful for building blocks that are not in the Enamine database
    with open('/home/ubuntu/synflownet/src/synflownet/data/building_blocks/pharmaron_input_bbs.txt', 'w') as f:
        for mol in suppl:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                f.write(smiles + '\n')
print("Done!")