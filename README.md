# Flappy Bird AI (from Scratch)

A minimal AI project where neural networks learn to play Flappy Bird using a genetic algorithm — all built from scratch, no ML libraries.

## Features
- Custom neural network (feedforward, numpy)
- Simple genetic algorithm (selection, crossover, mutation)
- Dynamic Flappy Bird environment (random pipe gaps & spacing)
- Clean `go.py` script to train & run everything

## Files
- `FlappyEnv.py` – game environment (pygame)
- `genetic.py` – genetic algorithm logic
- `neural_network.py` – basic neural net engine
- `go.py` – main script to launch training

## Run

```bash
python go.py
```

## Requirements

- Python 3.7+
- `pygame`, `numpy`

Install:

```bash
pip install pygame numpy
```

## Fitness

Each bird:  
`score = pipes_passed * 1000 + frames_survived`

---

This project is also suitable for generating AI training timelapse videos.  
License: MIT
