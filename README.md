# Social Influence Prediction GNN

This project predicts whether a user will do an action (like adopting
a product or reposting something) based on what their friends/connections
did, using a simple Graph Neural Network.

## Files

- `data.py` - makes a fake social network and generates labels
- `model.py` - the GNN model (written from scratch using numpy)
- `main.py` - run this file, it does everything (train + test + save results)
- `requirements.txt` - packages you need to install
- `model_weights.npz` - the trained model, created after you run main.py

## How to run it

1. Install the packages:
```
pip install -r requirements.txt
```

2. Run the code:
```
python main.py
```

That's it. It will print out the results and save two files:
- `model_weights.npz` (the trained model)
- `training_curve.png` (a graph showing training progress)

## What it does

1. Creates a fake social network (since we didn't have access to a real one)
2. Simulates people influencing their friends to do something
3. Builds a small neighborhood graph around each person
4. Trains a GNN to predict if a person will do the action, based on their friends
5. Compares it against a simpler model (Logistic Regression) to show the GNN does better

## Results

| Model | Accuracy (AUC) |
|---|---|
| Logistic Regression | ~0.79 |
| Plain GCN | ~0.99 |
| Full model (GCN + extra features) | ~0.99 |

The GNN models do much better because they can actually look at what a
person's friends did, not just some summary numbers.

## Note

We used a fake/simulated dataset instead of a real one because we
couldn't access the original dataset. Everything else (model, training,
evaluation) is built for real and actually works.
