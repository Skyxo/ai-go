# PE50-AI-for-the-Game-of-GO ★ AlphaGo-Lite (9 × 9)

A compact, research-grade re-implementation of the **AlphaGo Zero** pipeline on a 9 × 9 board.

📄 **Project report (PDF)** → [docs/Rapport_PE_050_2024.pdf](docs/Rapport_PE_050_2024.pdf)  
*Section 5 (Neural Networks) and all technical appendices were written by **Charles Bergeat***.

The engine combines **Monte-Carlo Tree Search (MCTS)** with twin neural networks (policy + value).  
- **Bootstrap phase :** the networks are first trained on **30 000 KataGo-generated games** (no human records).  
- **Improvement phase :** the agent then refines itself through iterative **self-play**.

---

### 🎮 A Glimpse of the GUI

| Main launcher (board size & mode) | In-game 19 × 19 goban |
|---|---|
| <img src="https://github.com/user-attachments/assets/2aea3f62-bd35-46dd-abd2-c2f29c8ba73d" width="370" alt="Main menu"> | <img src="https://github.com/user-attachments/assets/da6dc5ab-a4ef-4c41-a0d1-904807f05c49" width="370" alt="19x19 board"> |

*From the menu you can spawn a **sandbox**, a **local 2-player** match, play vs the AI (black / white) or run **AI vs AI** benchmarks on 5×5, 9×9, 13×13 or 19×19 boards.*

---

## ✨ Key Features

| Axis                | Highlights                                                                                                                                                          |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Search / MCTS**   | PUCT selection, Dirichlet noise at the root, NN value evaluation. Default **2 500 sims** per move with **C<sub>PUCT</sub> = 1.1**.                                   |
| **Policy Network**  | 8-block ResNet with Squeeze-and-Excitation; dropout-optimised head; predicts 81 moves + pass in a single softmax.                                                    |
| **Value Network**   | 6 residual blocks with global attention; Sigmoid head yielding a win-probability in \[0–1].                                                                         |
| **Reinforcement Loop** | 50 self-play games → 25 training epochs → evaluation vs. previous net every cycle; CLI supports batching & multi-GPU.                                            |
| **Dataset Bootstrap** | 30 k 9×9 games produced by **KataGo** (ELO ≈ 14 k) → 17 M positions after 8-fold symmetry augmentation, giving a strong starting point.                            |
| **Interactive GUI** | Tkinter front-end (5×5 – 19×19): AI-vs-Human, AI-vs-AI, sandbox mode & optional heat-map overlay.                                                                    |

---

## 📂 Project Structure

```text
PE50-AI-for-the-Game-of-GO/
├── main.py                           # Tkinter GUI & game launcher
├── MCTS_GO.py                        # Core MCTS implementation
├── katago_relatives/
│   └── katago-opencl-linux/
│       └── networks/
│           ├── go_policy_9x9_v5.py   # Policy network
│           ├── go_value_9x9_v3.py    # Value network
│           └── reinforcement_learning.py  # Self-play + RL driver
├── plateau.py | noeud_go.py | arbre_go.py  # Board & tree helpers
├── moteursDeJeu.py | const.py              # Game engines & constants
├── ressource/                              # GUI assets
└── docs/
    ├── Rapport_PE_050_2024.pdf             # Project report (this PDF)
    └── plots/                              # Training curves
````

---

## ⚙️ Installation

```bash
# Clone & enter
git clone https://github.com/Skyxo/PE50-AI-for-the-Game-of-GO.git
cd PE50-AI-for-the-Game-of-GO

# Create a virtual env (Python ≥ 3.9 recommended)
python -m venv .venv && source .venv/bin/activate

# Core dependencies
pip install torch numpy matplotlib tqdm tkinter
```

*KataGo binaries* for data boot-strapping are vendored in `katago_relatives/`; you can swap in a newer network if desired.

---

## 🚀 Quick Start

### 1 · Pre-Training + Self-Play

```bash
python katago_relatives/katago-opencl-linux/networks/reinforcement_learning.py \
  --iterations 100 \         # RL cycles
  --games 50 \               # self-play games per cycle
  --epochs 25 \              # NN epochs per cycle
  --mcts_sims 800 \          # simulations per move
  --eval_simulations 3200 \  # sims for evaluation
  --eval_games 20            # evaluation matches per cycle
```

During the **first** cycle the script loads the KataGo dataset and performs supervised pre-training on policy & value heads before switching to pure self-play.
Resume a run with `--resume checkpoints/latest.pth`.

### 2 · Play Through the GUI

```bash
python main.py
```

* Choose board size (5 / 9 / 13 / 19).
* Launch **Sandbox**, **Human vs Human**, **Play vs AI**, or **AI vs AI**.
* Toggle the heat-map overlay to visualise MCTS visits.

---

## 📊 Training & Evaluation Results

[cite_start]To build a competent AI for the game of Go, our system relies on Monte Carlo Tree Search (MCTS) guided by two distinct neural networks: a Policy Network to propose promising moves, and a Value Network to evaluate the board state[cite: 307]. [cite_start]The training process is divided into an initial supervised learning phase (using datasets generated via KataGo) followed by a self-play reinforcement learning phase[cite: 377, 378, 666].

Below is the chronological breakdown of the training dynamics and the final evaluation.

### 1 · Policy Network Convergence (Entropy)

[cite_start]**Context:** The Policy Network's job is to predict the probability distribution over all possible legal moves from a given board state[cite: 317, 388]. [cite_start]To monitor its stability during the supervised training phase, we track its prediction *entropy*, which measures the dispersion of probabilities[cite: 516, 520]. [cite_start]High entropy means the model is hesitant (uniform distribution), while low entropy means it is highly confident[cite: 520].

<img src="https://github.com/user-attachments/assets/d438a4ef-0da5-4350-abb1-64191c79b077" width="720" alt="Policy entropy">

* [cite_start]**Steady Confidence Growth:** The entropy smoothly falls from an initial ~3.5 nats down to ~2.2 nats[cite: 551, 552, 554]. [cite_start]This indicates that the network is becoming increasingly confident in its predictions without overfitting or becoming overly rigid[cite: 552, 554, 555].
* [cite_start]**Optimizer Refresh:** The vertical green lines mark deliberate `refresh_model` events where the optimizer's momentum is reset[cite: 550, 553]. [cite_start]Notice that these resets do not disrupt the entropy, proving that the model's confidence remains stable even when escaping optimization plateaus[cite: 553].

### 2 · Value Network Learning Curve

[cite_start]**Context:** While the Policy Network suggests *what* to play, the Value Network evaluates *who is winning* by predicting the current player's probability of victory (a scalar between 0 and 1)[cite: 328, 329]. [cite_start]We track this using Mean Squared Error (MSE) and the R² coefficient of determination to ensure the predictions accurately reflect the true game outcomes[cite: 602, 603].

<img src="https://github.com/user-attachments/assets/5550562e-4ab5-4471-b521-850d9a8003e5" width="720" alt="Value-net loss & R²">

* [cite_start]**Top (MSE):** The training (blue) and validation (red) losses decline steadily before stabilizing around 0.05 after epoch 150[cite: 656]. [cite_start]The periodic green lines (optimizer refresh) and grey lines (learning rate scheduler drops) effectively push the network out of local minima to refine its accuracy[cite: 657]. 
* [cite_start]**Bottom (R²):** The coefficient of determination climbs from ~0.58 and plateaus at ~0.75[cite: 658]. [cite_start]This means the Value Network successfully captures 75% of the target variance, providing highly reliable board evaluations for the MCTS without showing signs of overfitting[cite: 658, 659].

### 3 · Reinforcement Learning via Self-Play

[cite_start]**Context:** After the supervised learning phase, the agent is trained through self-play (playing against its own previous iterations) to discover new strategies without human data[cite: 666, 669]. [cite_start]Each iteration consists of generating 50 new games (approx. 3,000 board positions), updating the networks over 25 epochs, and evaluating the new model[cite: 672, 673, 677, 776, 777]. 

<img src="https://github.com/user-attachments/assets/e53fb909-7ab4-42e6-863c-4b0a7056e906" width="720" alt="Policy & Value losses by iteration">

* [cite_start]**Policy Loss (Left):** Over 9 iterations (moving from dark purple to yellow), the Kullback-Leibler (KL) divergence drops significantly[cite: 779]. [cite_start]Iteration 1 starts at 3.3 and finishes at 2.6, while by Iteration 9, the loss starts and stays closer to 1.9[cite: 780, 781]. [cite_start]This shows the policy is aligning faster and more accurately with the targets generated by the MCTS[cite: 781].
* [cite_start]**Value Loss (Right):** A similar rapid improvement is visible[cite: 782]. [cite_start]The starting loss drops from 0.40 in the first iteration down to a stable 0.23–0.24 plateau by iterations 8 and 9[cite: 782]. [cite_start]The Value network successfully scales its precision alongside the increasing complexity of the self-play games[cite: 783].

### 4 · Final Evaluation: Neural Agent vs. Classic MCTS

[cite_start]**Context:** To definitively prove the efficiency of the neural integration, we pitted our final Neural MCTS against a "classic" heuristic-only MCTS baseline (which relies on random rollouts and no neural networks)[cite: 846]. [cite_start]To ensure a fair comparison, both agents were restricted to an identical computation budget of exactly 800 simulations per move[cite: 846, 847]. 

| **Neural agent vs. pure MCTS** (100-game match-up) |
|----------------------------------------------------|
| <img src="https://github.com/user-attachments/assets/a12eb50e-accd-4b43-becf-45df975961dc" width="650" alt="Win-rate histogram"> |

* [cite_start]**Crushing Superiority:** The neural agent (green) achieved a massive **96% win rate** over 100 games[cite: 849]. [cite_start]It secured 47 wins out of 50 while playing Black, and 49 out of 50 while playing White[cite: 849].
* [cite_start]**Conclusion:** This proves that replacing random rollouts with a Value Network and guiding tree expansion with a Policy Network drastically increases the quality of the search[cite: 850, 851]. [cite_start]At an equal simulation budget, the neural architecture makes vastly superior decisions[cite: 852].

---

## 🔬 Research Internals

* [cite_start]**Minimal Input Planes:** To keep training fast and force the network to learn genuine tactics (rather than relying on pre-calculated heuristics), the board state is simplified into just 2 binary channels: current player stones and opponent stones[cite: 318, 381].
* [cite_start]**Top-k Expansion:** During the MCTS expansion phase, the algorithm is restricted to exploring only the top *k* most probable moves suggested by the Policy Network, effectively managing the massive search space of Go and keeping tree growth in check[cite: 360, 362].
* [cite_start]**Optimizer Refresh Mechanism:** We implemented a custom `refresh_model` routine that periodically resets the momentum buffer of the RAdam optimizer without altering the network weights[cite: 454, 500]. [cite_start]This frees the model from accumulated momentum that can "freeze" attention coefficients, allowing it to rapidly escape optimization plateaus[cite: 501, 503, 504].


---

## 🛣️ Roadmap

* [ ] Curriculum learning for 13×13 then 19×19
* [ ] SGF export + GTP bridge
* [ ] Distributed self-play (Ray / MPI)
* [ ] Heat-map overlay inside the GUI
* [ ] CI pipeline & pre-commit hooks

---

## 🤝 Contributing

1. **Fork** → create a feature branch (`feat/your-feature`).
2. Format with `black` / `flake8`, run tests if you add them.
3. Open a detailed Pull Request (screenshots welcome!).

---

## 📜 Credits & Licence

*Built by the PE-50 team (École Centrale de Lyon, 2025).*
Inspired by **DeepMind’s AlphaGo Zero** and the open-source **KataGo** community.
Code released under the **MIT Licence** — see `LICENSE` for details.




