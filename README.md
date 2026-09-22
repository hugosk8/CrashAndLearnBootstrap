# Crash & Learn — Bootstrap RL (CartPole-v1)

Journal de progression du bootstrap RL (Epitech Crash & Learn). Objectif : comprendre les briques de base du RL (agent, environnement, observation, action, reward, Q-function, epsilon-greedy, replay buffer, target network) via CartPole-v1, avant d'attaquer le projet de voiture autonome sur simulateur F1TENTH.

Référence : `CrashAndLearn-bootstrap.pdf` (les 4 steps et leurs questions), `INSTRUCTIONS.md`/`INSTRUCTIONS.pdf` (setup technique).

---

## Step 1 — Observer une politique aléatoire

- [x] Lire la doc CartPole (via observation directe + doc gymnasium)
- [x] Lancer `eval.py` avec `USE_RANDOM=True`, observer 5 épisodes
- [ ] Print de toutes les actions/observations/rewards/terminations à chaque step (fait partiellement via `print(obs)` en fin d'épisode)

### Questions du PDF

- [x] **Quand un épisode se termine-t-il ?**
  `terminated=True` : condition d'échec physique atteinte (angle du pole > ~12°, ou chariot hors de la zone ±2.4). `truncated=True` : limite artificielle de durée atteinte (500 steps), sans rapport avec un échec.

- [ ] **Qu'est-ce qu'une observation ?** *(pas encore testé : perturber une feature et observer la stabilité)*
  Vecteur de 4 valeurs : `[position du chariot, vitesse du chariot, angle du pole, vitesse angulaire du pole]`.

- [x] **Qu'est-ce qu'une action ?**
  2 actions possibles : `0` = pousser le chariot à gauche, `1` = pousser à droite.
  Expérience : remplacer la politique random par une règle déterministe basée sur l'angle (`action = 1 if obs[2] > 0 else 0`, intuition du balai qu'on rattrape dans le sens où il penche).
  Résultat : règle nettement meilleure que le hasard (~37-40 de reward moyen vs ~13-20 en random, sur plusieurs runs de 5 épisodes, aucun chevauchement entre les deux groupes).

  **Expérience bruit** (`noise_experiment.py`) : mélange règle/random contrôlé par une probabilité `p` tirée à chaque step (`random() <= p` → action random, sinon règle perso) — prévisualisation du concept d'epsilon-greedy vu plus tard en Step 3.
  - Méthodologie : pour chaque `p` dans `[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]`, 5 runs indépendants de 30 épisodes chacun. `run_episodes(env, p, 30)` retourne la moyenne d'un run ; `main()` répète ça 5 fois par `p` et calcule `mean_rewards`/`std_rewards` (moyenne et écart-type **sur les 5 runs**, pas sur les épisodes) pour quantifier le bruit inter-runs observé en relançant le script à la main.
  - Résultat (ordre de grandeur, varie légèrement d'un lancement à l'autre car l'env n'est pas seedé) :

    | p | mean | std |
    |---|------|-----|
    | 0.00 | ~41-43 | ~1-2 |
    | 0.10 | ~42-44 | ~1-4 |
    | 0.25 | ~45-48 | ~3-4 |
    | 0.50 | ~39-42 | ~2-4 |
    | 0.75 | ~32-33 | ~2-3 |
    | 0.90 | ~25-28 | ~3 |
    | 1.00 | ~21-22 | ~1-2 |

  - Méthode de comparaison : fourchette `mean ± std` par `p` ; si deux fourchettes ne se chevauchent pas, la différence est probablement réelle (pas juste du bruit inter-runs).
  - Conclusion : dégradation **ni progressive dès le début, ni un seuil unique brutal** — plutôt un **plateau de tolérance** jusqu'à `p≈0.5` (fourchettes qui se chevauchent, pas de différence significative avec `p=0`), suivi d'une **chute nette** entre `p=0.5` et `p=0.75` (fourchettes disjointes) jusqu'à `p=1` (≈ performance random pure, ~22). Hypothèse : la règle reste efficace tant que le bruit ne domine pas majoritairement la décision.
  - [x] Courbe visualisée avec matplotlib (`noise_experiment.png` : mean_rewards + barres d'erreur = std_rewards). Confirme visuellement l'analyse : plateau (léger pic même à p=0.25) jusqu'à p≈0.5, puis décroissance assez régulière jusqu'à p=1 plutôt qu'un seuil brutal isolé.
  - [x] Expérience complète relancée 5 fois pour comparer la variabilité d'une expérience à l'autre (`noise_experiment_5runs.png`, fonction `experiment()` extraite pour être répétée). Les 5 courbes ont la même forme générale (plateau/léger pic vers p=0.1-0.25, chute jusqu'à ~22-23 à p=1) — la variabilité inter-expériences est surtout visible dans la zone de transition p=0.5-0.9, ce qui confirme que le comportement observé n'est pas un artefact d'une seule run chanceuse.

- [ ] **Qu'est-ce que la reward ?** *(pas encore testé : modifier la reward function, ex: -1 si échec / 0 sinon, et observer l'effet)*

### Points appris en cours de route (utiles pour la suite)

- Bug corrigé dans `eval.py` : chargement du modèle (`load_state_dict`) se faisait même avec `USE_RANDOM=True` → `FileNotFoundError`. Corrigé en entourant d'un `if not USE_RANDOM:`.
- `set_seed(42)` (dans `utils.py`) fixe les générateurs `random`, `np.random`, `torch` — **mais pas** le générateur aléatoire interne à l'environnement Gymnasium (`env.reset()`, `env.action_space.sample()`). D'où des résultats qui varient d'un lancement de script à l'autre malgré la seed fixée.
- Un `tensor` PyTorch = structure de données similaire à un tableau, optimisée pour le calcul matriciel et la différentiation automatique. Les réseaux attendent une dimension de "batch" en plus des features (`obs` de forme `(4,)` → `[obs]` de forme `(1, 4)`).
- Les hyperparamètres d'architecture (`obs_dim`, `action_dim`, `hid_size`) dans `eval.py` doivent correspondre exactement à ceux utilisés à l'entraînement, sinon le chargement des poids sauvegardés échoue.
- Avec peu d'épisodes (5), les moyennes de reward sont bruitées à cause du point de départ aléatoire du pole — prévoir un échantillon plus grand (30-50+) pour des comparaisons fiables.
- En Python, `variable = valeur` **remplace** tout le contenu d'une liste ; `liste.append(valeur)` **ajoute** un élément sans écraser les précédents. Piège rencontré deux fois de suite dans `noise_experiment.py` (une fois sur les rewards par épisode, une fois sur les moyennes par `p`).
- Une fonction appelée avant sa définition (ou définie par erreur *à l'intérieur* d'une autre fonction, donc invisible ailleurs) provoque une `NameError` — Python exécute un fichier séquentiellement, de haut en bas ; une fonction imbriquée n'existe que pendant l'exécution de la fonction qui la contient.
- Pour comparer deux moyennes bruitées calculées sur peu de répétitions (sans test statistique formel type t-test) : comparer les fourchettes `mean ± std` — si elles ne se chevauchent pas, la différence est probablement réelle plutôt que du bruit d'échantillonnage.

---

## Step 2 — Lancer un DQN fonctionnel

- [ ] Entraîner l'agent (`python3 train_cartpole.py`)
- [ ] Visualiser la courbe d'apprentissage (TensorBoard : `tensorboard --logdir runs/cartpole_dqn`)
- [ ] Observer l'agent entraîné (`eval.py` avec `USE_RANDOM=False`)

### Questions du PDF

- [ ] Pourquoi la performance s'améliore-t-elle ?
- [ ] Pourquoi l'exploration est-elle nécessaire ? Comment est-elle faite ici ?

---

## Step 3 — Lire l'algorithme (`dqn.py`)

- [ ] Où les observations entrent dans le réseau
- [ ] Où le système "décide" des actions
- [ ] Où les rewards sont stockées
- [ ] Où le réseau de neurones est mis à jour
- [ ] Où le target network est utilisé
- [ ] Où l'exploration epsilon-greedy se produit

---

## Step 4 — Petites modifications (hyperparamètres)

- [ ] Learning rate
- [ ] Discount factor (gamma)
- [ ] Taille du replay buffer
- [ ] Exploration schedule
- [ ] Taille du réseau
- [ ] Batch size

Discuter : qu'est-ce qui casse l'entraînement en premier ? Comment accélérer l'entraînement (multiprocessing, subproc env...) ?
