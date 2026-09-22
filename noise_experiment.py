import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random as pyrandom
import gymnasium as gym
import matplotlib.pyplot as plt
import statistics

# Joue un épisode en mélangeant la règle perso (angle du pole) et du random,
# selon une probabilité p : p=0 -> toujours la règle, p=1 -> toujours random
# Prévisualise le compromis exploitation/exploration (epsilon-greedy)
def run_episodes(env, p, n_episodes):
    rewards_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rewards = 0.0
        step_idx = 0

        while not (done or truncated):
            if pyrandom.random() <= p:
                action = env.action_space.sample()
            else:
                # Règle perso : pousser dans le sens où le pole penche (obs[2] = angle)
                action = 1 if obs[2] > 0 else 0

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated
            ep_rewards += reward
            step_idx += 1

        rewards_list.append(ep_rewards)

    # Reward moyenne d'UN run de n_episodes épisodes
    return sum(rewards_list) / len(rewards_list)

# Pour chaque valeur de p, refait num_runs runs indépendants (chacun de num_episodes
# épisodes) et calcule la moyenne/écart-type ENTRE ces runs. Le std ici mesure donc
# le bruit inter-runs (l'env n'est pas seedé), pas la variance episode par episode
def experiment(env, p_values, num_episodes, num_runs):
    mean_rewards = []
    std_rewards = []

    for p in p_values:
        runs_results = []

        for _ in range(num_runs):
            avg = run_episodes(env, p, num_episodes)
            runs_results.append(avg)

        mean_rewards.append(statistics.mean(runs_results))
        std_rewards.append(statistics.stdev(runs_results))

    return mean_rewards, std_rewards

def main():
    p_values = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    env = gym.make("CartPole-v1", render_mode = None)
    num_episodes = 30
    num_runs = 5
    # Nombre de fois où on répète l'expérience COMPLÈTE (tous les p_values), pour
    # comparer visuellement si la forme de la courbe est stable d'une fois à l'autre.
    num_experiments = 5

    for exp_idx in range(num_experiments):
        mean_rewards, std_rewards = experiment(env, p_values, num_episodes, num_runs)

        print(f"--- Expérience {exp_idx + 1} ---")
        for p, mean, std in zip(p_values, mean_rewards, std_rewards):
            print(f"p={p:.2f}  mean={mean:.2f}  std={std:.2f}")

        # Une courbe par expérience, superposées sur le même graphe.
        plt.plot(p_values, mean_rewards, marker="o", label=f"Expérience {exp_idx + 1}")

    env.close()

    plt.xlabel("p (probabilité d'action random)")
    plt.ylabel("Reward moyenne (sur 5 runs de 30 épisodes)")
    plt.title("Variabilité de la courbe d'une expérience à l'autre")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("noise_experiment_5runs.png")
    plt.show()

if __name__ == "__main__":
    main()
