"""Run PPO training for Pokemon TCG agent with longer training and evaluation."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from env_wrapper import CabtEnv
from train import PPO
from main import DECK

OBS_DIM = 61


def main():
    print("Initializing environment...")
    env = CabtEnv(DECK, DECK.copy(), max_episode_steps=1000)

    print("Initializing PPO...")
    ppo = PPO(
        obs_dim=OBS_DIM,
        action_dim=64,
        hidden_dim=256,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        update_epochs=4,
        max_grad_norm=0.5,
        device="cpu",
    )

    print("Starting training (200k steps)...")
    agent = ppo.train(
        env=env,
        total_steps=200_000,
        rollout_steps=1024,
        log_interval=5,
        eval_interval=30,
        eval_episodes=20,
        save_path="checkpoints/model.pt",
    )

    print("\n=== Final Evaluation ===")
    final_eval = ppo.evaluate(env, n_episodes=50)
    print(f"Win rate: {final_eval['win_rate']:.3f}")
    print(f"Avg reward: {final_eval['avg_reward']:.3f}")

    agent.save("model.pt")
    print("Model saved to model.pt (for submission)")
    env.close()


if __name__ == "__main__":
    main()
