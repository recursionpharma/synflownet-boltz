# Additional implementation notes

This repository is centered around training GFlowNets that produce molecules from sequentially applying chemical reactions to reactants (building blocks). The building blocks were made available by Enamine upon request and the reaction templates are modified Hartenfeller-Button reaction templates.

### Environment, Context, Trainer

We separate experiment concerns in distinct classes:

- `ReactionTemplateEnv` defines an MDP which starts from an empty graph, followed by an Enamine building block. Stepping forward in the environment consists in running a reaction using RDKit.
- `ReactionTemplateEnvContext` provides an interface between the agent and the environment, it:
    - maps graphs to torch_geometric `Data`
  instances.
    - maps GraphActions to action indices.
    - produces action masks.
    - communicates to the model what inputs it should expect.
- The `RewardQueue` and `PersistentReplayBuffer` classes are responsible for storing and retrieving trajectories and rewards which are used to train the model.
- The `AsyncRewardTrainer` class is responsible for instantiating everything, and running the training loop.

### Data

The training relies on two data sources: reaction templates and building blocks, which can be found in `synflownet/data/`. The model uses pre-computed masks to ensure compatibility between the building blocks and the reaction templates. If absent, these masks will be re-computed when initiating a new training run. The data for Boltz-2 inference i.e. the protein sequence and msa files, can be found in `synflownet-boltz-launcher/data`.

### Policies and action categoricals

The `GraphTransformerSynGFN` class is used to parameterize the policies and outputs a specific categorical distribution type for the actions defined in `ReactionTemplateEnvContext`. If `config.model.graph_transformer.continuous_action_embs` is set to `True`, then the probability of sampling building blocks is computed from the normalized dot product of the molecule representation and the embedding vector of the state. The `ActionCategorical` class contains the logic to sample from the hierarchical distribution of actions.