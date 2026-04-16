# Machine Learning in Finance

Coursework for the *Machine Learning in Finance* course (WU Wien). The repository is organized week by week and contains homework solutions (Team / Group 3) together with the lecture example scripts used as starting points.

## Contents

- **Week 1** — Introduction and first homework.
  - [Exercise_1.ipynb](Week%201/Exercise_1.ipynb)
  - [HW1_team3.ipynb](Week%201/HW1_team3.ipynb)
- **Week 2** — Gradient descent on non-smooth objectives `|x - b|^p` with `b ~ N(1,1)`.
  - [week2_group3.ipynb](Week%202/week2_group3.ipynb)
- **Week 3** — European option pricing under Black–Scholes, CEV and Heston models via neural-network pricers.
  - [ExampleW3_pricer.py](Week%203/ExampleW3_pricer.py) — lecture example.
  - [HW3_Q2_solution_team3_vscode_math.ipynb](Week%203/HW3_Q2_solution_team3_vscode_math.ipynb)
  - [Week 3.py](Week%203/Week%203.py)
- **Week 4** — Deep Hedging: learning hedging strategies for call and barrier options, including a constrained variant.
  - [ExampleW4_DeepHedging.py](Week%204/ExampleW4_DeepHedging.py) — lecture example.
  - [Week 4.py](Week%204/Week%204.py)
  - [ML_week4.pdf](Week%204/ML_week4.pdf), [week4_ex1_barrier.png](Week%204/week4_ex1_barrier.png), [week4_ex2_constrained.png](Week%204/week4_ex2_constrained.png)
- **Week 5** — Signature methods: recovering volatility from a signature model `X_t = c0 + c1·W_t + c2·(W_t² − t)/2`.
  - [Week 5.ipynb](Week%205/Week%205.ipynb)
  - [Week 5.py](Week%205/Week%205.py)

## Environment

A local virtual environment lives in [.venv/](.venv/). To recreate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib tensorflow keras jupyter
```

The Week 3 and Week 4 example scripts require `tensorflow` / `keras`; the remaining notebooks only need the standard scientific Python stack.

## Usage

Open the workspace file [Machine Learning in Finance.code-workspace](Machine%20Learning%20in%20Finance.code-workspace) in VS Code, select the `.venv` interpreter, and run any week's notebook or script directly.
