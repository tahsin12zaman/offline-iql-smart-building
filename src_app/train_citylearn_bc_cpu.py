import argparse
from pathlib import Path
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import BCConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/citylearn_logged_multi.npz")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data)
    arr = np.load(data_path, allow_pickle=True)

    observations = arr["observations"].astype(np.float32)
    actions = arr["actions"].astype(np.float32)
    rewards = arr["rewards"].astype(np.float32)
    terminals = arr["terminals"].astype(bool)

    print("=" * 70)
    print("CityLearn Offline RL Training: BC")
    print("=" * 70)
    print("Data:", data_path)
    print("Observations:", observations.shape)
    print("Actions:", actions.shape)
    print("Rewards:", rewards.shape)
    print("Terminals:", terminals.shape)
    print("Steps:", args.steps)
    print("Seed:", args.seed)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    algo = BCConfig(
        batch_size=256,
        learning_rate=1e-3,
    ).create(device=False)

    algo.fit(
        dataset,
        n_steps=args.steps,
        n_steps_per_epoch=1000,
        show_progress=True,
    )

    model_path = f"results/citylearn_bc_seed_{args.seed}.d3"
    Path("results").mkdir(exist_ok=True)
    algo.save_model(model_path)

    print("=" * 70)
    print("BC training finished.")
    print("Saved model:", model_path)
    print("=" * 70)


if __name__ == "__main__":
    main()