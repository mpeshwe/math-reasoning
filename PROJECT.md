# Math Reasoning Distillation with Teacher-Guided GRPO

## Project Overview

Train a small language model (Llama 3.2 3B) to solve math reasoning problems
by combining GRPO reinforcement learning with a conditional meta-teacher (Llama 3.1 70B)
that intervenes only when the student model fails completely.

**Target benchmark:** GSM8K (Grade School Math 8K)
**Target improvement:** ~20% accuracy gain on the 3B student model
**Infrastructure:** Python, Ray, Unsloth, Vast.ai GPU instances

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │              Ray Coordinator             │
                         └──────┬──────────┬──────────────┬────────┘
                                │          │              │
                       ┌────────▼───┐  ┌───▼────────┐  ┌─▼──────────────┐
                       │  Student   │  │  Teacher   │  │ Reward Model   │
                       │  3B       │  │  70B       │  │ (PRM)          │
                       │           │  │            │  │                │
                       │  Trains   │  │  Inference │  │ Inference      │
                       │  (LoRA)   │  │  only      │  │ only           │
                       │           │  │            │  │                │
                       │  GPU 1    │  │  GPU 0     │  │ GPU 1 (shared) │
                       └───────────┘  └────────────┘  └────────────────┘
```

### The Core Training Loop

```
For each batch of math problems:

    1. Student generates 8 solutions per problem
    2. Reward model (PRM) scores each solution step-by-step
    3. Check: are ANY solutions correct?

       YES ──→ Standard GRPO update
                (reinforce good solutions, discourage bad ones)

       NO  ──→ Teacher intervention
                - Teacher examines student's failed attempts
                - Teacher provides a targeted hint (not the answer)
                - Student retries with the hint
                - GRPO update on the retry attempts

    4. Update student weights
    5. Log metrics (accuracy, reward, teacher intervention rate)
    6. Repeat
```

---

## Models

| Role             | Model               | Size | Precision    | GPU         |
|------------------|----------------------|------|--------------|-------------|
| Student          | Llama 3.2 3B        | 3B   | FP16 + LoRA (Unsloth)  | GPU 1       |
| Teacher          | Llama 3.1 70B       | 70B  | 4-bit (AWQ)            | GPU 0       |
| Reward (PRM)     | Math-Shepherd (or custom) | 3B | FP16              | GPU 1       |

**Why same family (Llama)?** Same tokenizer = teacher feedback maps directly
to student vocabulary. No translation layer needed.

## Datasets

| Dataset      | Size   | Purpose                                    |
|--------------|--------|--------------------------------------------|
| GSM8K        | 8.5K   | Evaluation ONLY (never train on this)      |
| MetaMathQA   | ~395K  | Primary training data (augmented math)     |
| MATH         | ~12.5K | Harder problems for advanced training      |
| Synthetic    | varies | Teacher-generated solutions for SFT        |

## Key Libraries

| Library    | Purpose                                      |
|------------|----------------------------------------------|
| Unsloth    | 2-5x faster training, 70% less VRAM          |
| TRL        | SFT and GRPO trainers (works with Unsloth)   |
| Ray        | Multi-model coordination across GPUs          |
| vLLM/SGLang| Fast inference serving for teacher 70B        |
| wandb      | Training metrics logging and dashboards       |

### Unsloth

Unsloth optimizes LoRA training with custom CUDA kernels. Drop-in replacement
for HuggingFace model loading — same accuracy, much faster, much less memory.

```
Without Unsloth:                    With Unsloth:
    SFT on 3B: ~3-4 hrs on A100        SFT on 3B: ~1-1.5 hrs on A100
    VRAM: ~60-70 GB                     VRAM: ~20-30 GB
    Needs A100 80GB                     Can use A100 40GB or even RTX 4090

Unsloth handles: training (SFT, GRPO, DPO)
vLLM/SGLang handles: inference (teacher serving)
They solve different problems — use both.
```

## Hardware (Vast.ai)

With Unsloth's memory savings, cheaper GPUs become viable:

- Phase 1-3: 1x A100 40GB or RTX 4090 (~$0.50-1/hr)
- Phase 4-8: 1x A100 80GB (teacher) + 1x A100 40GB/RTX 4090 (student) (~$1.50-3/hr)
- Total estimated cost: $30-60

---

## Phase 0: Understand the Foundations

**Goal:** Build mental models of every concept before writing any code.
**Duration:** ~1 week
**Cost:** $0

### What to study

#### 1. Supervised Fine-Tuning (SFT)
The simplest form of training. You show the model correct answers and it
learns to mimic them. Like giving a student a textbook of worked examples.

```
Input:  "Janet's ducks lay 16 eggs per day..."
Target: "She eats 3, bakes 4, sells 9 remaining. 9 * 2 = $18"

Model reads input, predicts each token of target one at a time.
Loss = how far off each prediction was.
Weights adjust to reduce that loss.
Repeat across thousands of examples.
```

**Limitation:** The model learns to copy patterns, not truly reason.
It may memorize solutions rather than understanding the process.

#### 2. GRPO (Group Relative Policy Optimization)
A reinforcement learning method. Instead of showing correct answers, you
let the model try multiple times and learn from its own attempts.

```
Problem: "Janet's ducks lay 16 eggs..."

Student generates 8 attempts:
    Attempt 1: wrong    → score 0.0
    Attempt 2: correct  → score 1.0  (above average)
    Attempt 3: wrong    → score 0.0
    Attempt 4: correct  → score 1.0  (above average)
    Attempt 5: wrong    → score 0.1
    Attempt 6: wrong    → score 0.0
    Attempt 7: partial  → score 0.4
    Attempt 8: wrong    → score 0.0

Average score: 0.31

GRPO says:
    Attempts above average → make more likely (reinforce)
    Attempts below average → make less likely (discourage)

The model learns from its OWN successes, not from external answers.
```

**Key advantage over SFT:** The model develops its own reasoning strategies
rather than copying someone else's.

**Key weakness:** When ALL attempts fail, there's nothing good to reinforce.
This is exactly what the teacher intervention solves.

#### 3. Process Reward Models (PRM)
A model that scores each STEP of a solution, not just the final answer.

```
Why not just check the final answer?

    Solution: "16 - 3 = 12, 12 - 4 = 8, 8 * 2 = 16"
    Final answer: 16 (wrong, should be 18)

    But WHERE did it go wrong? Step 1: 16 - 3 = 12 (wrong!)

PRM scores:
    Step 1: "16 eggs total"           → 1.0 (correct)
    Step 2: "eats 3, leaving 12"      → 0.0 (WRONG: 16-3=13)
    Step 3: "bakes 4, leaving 8"      → 0.0 (wrong, cascading)
    Step 4: "8 * 2 = $16"             → 0.0 (wrong, cascading)

This step-level signal is much richer for training.
The student learns exactly WHERE its reasoning breaks.
```

#### 4. LoRA (Low-Rank Adaptation)
A memory-efficient training method. Instead of updating all 3B parameters,
you freeze the base model and train small adapter layers (~10M parameters).

```
Without LoRA:
    Update all 3B parameters
    Needs ~24 GB just for optimizer states
    Barely fits on one GPU with the model loaded

With LoRA:
    Freeze 3B parameters (read-only)
    Add tiny adapter matrices (~10M parameters)
    Only train adapters
    Needs ~2 GB for optimizer states
    Plenty of room on one GPU

    Results are 90-95% as good as full fine-tuning.
```

#### 5. Unsloth
A library that makes LoRA training 2-5x faster and uses 70% less memory.
It replaces HuggingFace's model loading with optimized CUDA kernels.

```
Without Unsloth (standard HuggingFace):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    # Slow training, high VRAM usage

With Unsloth (drop-in replacement):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained("meta-llama/Llama-3.2-3B")
    # Same model, same LoRA, same TRL trainers
    # But 2-5x faster, 70% less VRAM, no accuracy loss

The rest of your code stays identical — SFTTrainer and GRPOTrainer
from TRL work directly with Unsloth models.
```

**Where Unsloth applies:**
- Phase 3 (SFT) — faster, cheaper training
- Phase 7 (GRPO) — fits student + reward model comfortably on one GPU

**Where Unsloth does NOT apply:**
- Teacher inference (Phase 5) — use vLLM/SGLang instead
- Reward model inference (Phase 6) — standard HuggingFace is fine

#### 6. Quantization
Compressing model weights to use less memory. The teacher model (70B) is
too large for one GPU at full precision, so we quantize it.

```
Full precision (FP16):  70B × 2 bytes = 140 GB   (doesn't fit)
4-bit quantized (AWQ):  70B × 0.5 bytes = ~35 GB (fits on A100 80GB)

The teacher only does inference (not training), so the small quality
loss from quantization is acceptable.
```

#### 7. Ray
A framework for running multiple processes across multiple GPUs.
Each model lives in its own Ray "actor" (a persistent process).
Ray handles communication, GPU assignment, and crash recovery.

```
Without Ray:
    Manual multiprocessing, shared memory, GPU pinning → fragile

With Ray:
    TeacherActor(num_gpus=1)  → automatically gets GPU 0
    StudentActor(num_gpus=1)  → automatically gets GPU 1

    teacher.generate.remote(prompts)  → runs async on GPU 0
    student.train.remote(batch)       → runs async on GPU 1

    ray.get([result1, result2])       → wait for both
```

### Resources to study

- GRPO paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
  in Open Language Models" (the paper that introduced GRPO)
- rStar-Math paper: "rStar-Math: Small LLMs Can Master Math Reasoning
  with Self-Evolved Deep Thinking" (Microsoft Research)
- HuggingFace TRL documentation: GRPO trainer section
- Unsloth documentation + GitHub wiki (setup, supported models, TRL integration)
- Ray documentation: Core concepts + Actors tutorial
- LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models"
- GSM8K paper: "Training Verifiers to Solve Math Word Problems"

### Checkpoint: You're ready for Phase 1 when you can explain

- [ ] What GRPO does differently from SFT
- [ ] Why a PRM is better than just checking the final answer
- [ ] Why LoRA exists and when to use it
- [ ] What a Ray actor is and why it helps with multi-model systems
- [ ] Why all 8 solutions being wrong is a problem for GRPO
- [ ] How the teacher intervention fixes that problem

---

## Phase 1: Data Pipeline

**Goal:** Download, explore, clean, and standardize the training/evaluation data.
**Duration:** ~1 week
**Cost:** $0 (local machine, no GPU)

### Directory structure to create

```
math-reasoning-distill/
├── pyproject.toml
├── configs/
│   └── vast_setup.sh
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_gsm8k.py
│   │   ├── load_metamath.py
│   │   └── normalize.py
│   └── eval/
│       ├── __init__.py
│       ├── gsm8k_eval.py
│       └── metrics.py
├── scripts/
│   └── explore_data.py
└── results/
    └── .gitkeep
```

### Tasks

#### 1.1 — Download and explore GSM8K

Load GSM8K from HuggingFace datasets. Understand the format:

```
{
    "question": "Janet's ducks lay 16 eggs per day...",
    "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs...\n#### 18"
}
```

Key things to figure out:
- How is the final answer marked? (the #### delimiter)
- What do the <<...>> annotations mean? (calculator annotations)
- How many steps does an average solution have?
- What's the distribution of difficulty?

#### 1.2 — Download and explore MetaMathQA

Same process. Understand how it augments GSM8K:
- Rephrased questions (same math, different wording)
- Backward questions ("if the answer is 18, what was the price?")
- Self-verification questions

#### 1.3 — Build normalization pipeline

Create a standardized format for all data:

```json
{
    "id": "gsm8k_001",
    "source": "gsm8k",
    "question": "Janet's ducks lay 16 eggs per day...",
    "solution_steps": [
        "Janet sells 16 - 3 - 4 = 9 duck eggs a day.",
        "She makes 9 * 2 = $18 every day."
    ],
    "final_answer": 18,
    "num_steps": 2
}
```

#### 1.4 — Build the evaluation function

This is the most critical piece. Must work perfectly:

```
Input:  model output string (messy, may contain text + math)
Output: extracted numerical answer

Then compare to ground truth.

Edge cases to handle:
    - "The answer is $18"        → 18
    - "#### 18"                  → 18
    - "18.00"                    → 18
    - "she makes 18 dollars"     → 18
    - "the total is -5"          → -5
    - "2/3"                      → 0.6667
```

#### 1.5 — Build metrics tracking

Functions to compute:
- Accuracy (% of correct final answers)
- Accuracy by number of reasoning steps (easier vs harder problems)

### Checkpoint: Phase 1 is done when

- [ ] GSM8K and MetaMathQA are downloaded and explored
- [ ] Normalization pipeline converts raw data to standardized format
- [ ] Evaluation function correctly extracts answers and computes accuracy
- [ ] You can run evaluation locally on mock model outputs
- [ ] All code tested on CPU with small samples

---

## Phase 2: Baseline Evaluation

**Goal:** Get the unmodified Llama 3.2 3B's score on GSM8K. This is the
number you're trying to beat.
**Duration:** 3-5 days
**Cost:** ~$2-3

### Vast.ai setup

Rent: 1x A100 80GB
Image: PyTorch 2.x + CUDA

```
vast_setup.sh should:
    - Install your project from git
    - Install dependencies (transformers, datasets, vllm, etc.)
    - Download Llama 3.2 3B
    - Run evaluation
    - Upload results
    - Print "DONE — shut me down"
```

**Rule: never write or debug code on a paid GPU instance.**
Develop locally, test on CPU with 5 problems, then run the full
evaluation on Vast.ai.

### Tasks

#### 2.1 — Write inference script

Load Llama 3.2 3B, generate solutions for GSM8K test set (1,319 problems).

Key decisions:
- What chat template / system prompt to use?
- Temperature: 0.0 (greedy) for baseline, 0.7 for sampling later
- Max tokens: ~512 is enough for GSM8K solutions
- Batch size: as large as GPU memory allows

#### 2.2 — Run evaluation

```
Expected result:
    Llama 3.2 3B Instruct on GSM8K: ~45% accuracy

    This is your floor. Everything you do should beat this.
```

#### 2.3 — Analyze errors

Look at 50-100 wrong answers manually. Categorize:
- Arithmetic errors (right approach, wrong calculation)
- Logic errors (wrong approach entirely)
- Formatting errors (right answer, can't parse it)
- Giving up (model says "I'm not sure")

This tells you what the student needs to learn.

#### 2.4 — Save everything

- Model outputs for every problem → results/baseline_outputs.jsonl
- Accuracy numbers → results/baseline_metrics.json
- Push to git
- SHUT DOWN THE INSTANCE

### Checkpoint: Phase 2 is done when

- [ ] You have a concrete baseline accuracy number
- [ ] You've manually reviewed errors and understand failure modes
- [ ] Results are saved to the repo
- [ ] Vast.ai instance is shut down

---

## Phase 3: Supervised Fine-Tuning (SFT)

**Goal:** First training run. Teach the student by showing it correct solutions.
**Duration:** 1-2 weeks
**Cost:** ~$5-10

### Concept

```
SFT is the simplest training approach:

    For each (problem, correct_solution) pair in MetaMathQA:
        1. Feed problem tokens to the model
        2. Model predicts next token of the solution
        3. Compute loss (how wrong was the prediction)
        4. Backpropagate to update LoRA weights

    After training on ~395K examples:
        Student accuracy: ~45% → ~55-60%
```

### Tasks

#### 3.1 — Set up training configuration

```yaml
# configs/train_sft.yaml
model_name: meta-llama/Llama-3.2-3B-Instruct
dataset: MetaMathQA
lora_rank: 64
lora_alpha: 128
learning_rate: 2e-5
batch_size: 4
gradient_accumulation_steps: 8  (effective batch = 32)
num_epochs: 2
max_seq_length: 1024
save_steps: 500
```

#### 3.2 — Implement SFT training script

Using Unsloth + TRL's SFTTrainer:
- Load base model + tokenizer via Unsloth's FastLanguageModel
- Apply LoRA configuration (Unsloth handles the optimized kernels)
- Load and format MetaMathQA dataset
- Train with SFTTrainer (same TRL API, Unsloth accelerates it)
- Save LoRA adapter weights
- Upload to HuggingFace Hub

#### 3.3 — Run training on Vast.ai

```
Training time estimate (with Unsloth):
    395K samples × 2 epochs = 790K steps / 32 batch = ~25K optimizer steps
    ~1-2 hours on A100 (was 3-4 hrs without Unsloth)
    Can also run on RTX 4090 in ~2-3 hours

    Cost: ~$1-3
```

#### 3.4 — Evaluate the SFT model

Load base model + LoRA adapter, run GSM8K evaluation.

```
Expected:
    Before SFT: ~45%
    After SFT:  ~55-60%

    This is your first real result.
```

#### 3.5 — Analyze what improved and what didn't

Compare errors before/after SFT:
- Did arithmetic errors decrease?
- Did logic errors decrease?
- Are there new error types?

### Checkpoint: Phase 3 is done when

- [ ] SFT training completed successfully
- [ ] LoRA adapter saved to HuggingFace Hub
- [ ] GSM8K accuracy improved over baseline
- [ ] Error analysis shows what improved
- [ ] Training curves (loss over steps) saved

---

## Phase 4: Learn Ray

**Goal:** Learn Ray by building the multi-model coordination system
with fake (mock) models on your local CPU.
**Duration:** ~1 week
**Cost:** $0

### Why fake models first?

```
Real models:
    - Need GPUs ($$$)
    - Slow to load (minutes)
    - Slow to generate (seconds per sample)
    - Hard to debug when something breaks

Fake models:
    - Run on CPU (free)
    - Instant to load
    - Instant to "generate" (return canned strings)
    - Easy to debug

    Build the coordination logic first.
    Swap in real models later.
```

### Tasks

#### 4.1 — Ray basics

Learn and practice:
- ray.init() — start the runtime
- @ray.remote — mark a class/function as remote
- .remote() — call it asynchronously
- ray.get() — wait for results
- Actors — classes that stay alive and hold state
- Resource declaration — num_gpus, num_cpus

#### 4.2 — Build FakeTeacher actor

```
class FakeTeacher:
    def analyze_and_hint(problem, failed_attempts):
        return "Hint: try subtracting before multiplying"
```

#### 4.3 — Build FakeStudent actor

```
class FakeStudent:
    def generate(problem, n=8):
        return [random math strings]

    def generate_with_hint(problem, hint, n=8):
        return [slightly better random math strings]

    def train_step(solutions, rewards):
        print("Would update weights here")
```

#### 4.4 — Build FakeRewardModel actor

```
class FakeRewardModel:
    def score(solutions):
        return [random scores between 0.0 and 1.0]
```

#### 4.5 — Wire the full loop

```
The complete coordination flow with fakes:

    for batch in problems:
        solutions = student.generate.remote(batch, n=8)
        rewards = reward_model.score.remote(solutions)

        if any_correct(rewards):
            student.grpo_update.remote(solutions, rewards)
        else:
            hint = teacher.analyze_and_hint.remote(batch, solutions)
            retry = student.generate_with_hint.remote(batch, hint, n=8)
            retry_rewards = reward_model.score.remote(retry)
            student.grpo_update.remote(retry, retry_rewards)

        log_metrics(step, rewards, teacher_intervened)
```

Run this on CPU. Verify:
- The loop completes without errors
- Metrics are logged correctly
- Teacher only intervenes when all scores are low
- GPU resources are declared (even if not used on CPU)

### Checkpoint: Phase 4 is done when

- [ ] Full training loop runs on CPU with fake models
- [ ] Ray actors communicate correctly
- [ ] Teacher intervention logic triggers at the right time
- [ ] Metrics logging works (accuracy, reward, intervention rate)
- [ ] You understand Ray well enough to swap in real models

---

## Phase 5: Teacher Setup

**Goal:** Get Llama 3.1 70B running as a fast inference server.
**Duration:** ~1 week
**Cost:** ~$5-10

### The teacher's role

The teacher NEVER trains. It only generates text (inference).
It needs to be:
- Fast (student is waiting for hints)
- Reliable (can't crash mid-training)
- Good at analyzing mistakes and giving useful hints

### Tasks

#### 5.1 — Load 70B with quantization

Use AWQ 4-bit quantization. The model compresses from 140GB to ~35GB,
fitting on a single A100 80GB.

Use vLLM or SGLang as the inference server — they handle batching,
KV cache management, and continuous batching automatically.

```
SGLang server:
    - Loads model once
    - Accepts requests over HTTP or Python API
    - Handles multiple concurrent requests
    - Much faster than raw HuggingFace generate()
```

#### 5.2 — Design the teacher prompt

This is critical. The teacher needs to:
1. See the problem
2. See what the student tried
3. Identify the specific mistake
4. Give a hint that helps WITHOUT giving the answer

```
System prompt for teacher:

    "You are a math tutor. A student attempted to solve the problem
     below but all attempts were incorrect.

     Analyze the student's attempts. Identify the most common mistake.
     Give a SHORT, SPECIFIC hint that addresses that mistake.

     Rules:
     - Do NOT give the full solution
     - Do NOT give the final answer
     - Point out WHERE the reasoning goes wrong
     - Suggest a specific strategy for the failing step

     Problem: {problem}

     Student's attempts:
     {attempts}

     Your hint (1-2 sentences):"
```

#### 5.3 — Test hint quality

Run the teacher on 100 problems where the student (from Phase 3) fails.
Manually evaluate:
- Are the hints specific enough?
- Do they avoid giving away the answer?
- Would a human student find them helpful?

Iterate on the prompt until hints are good.

#### 5.4 — Integrate into Ray

Replace FakeTeacher with real TeacherActor.
The actor wraps the SGLang/vLLM client.

```
TeacherActor:
    __init__: connect to SGLang server
    analyze_and_hint: send prompt, return hint string
```

### Checkpoint: Phase 5 is done when

- [ ] 70B loads and runs on a single A100 (quantized)
- [ ] SGLang/vLLM server is stable and fast
- [ ] Teacher prompt generates useful, non-spoiling hints
- [ ] TeacherActor integrates into Ray loop
- [ ] Tested on 100 real failure cases

---

## Phase 6: Reward Model (PRM)

**Goal:** Set up a model that scores reasoning steps, not just final answers.
**Duration:** ~1 week
**Cost:** ~$3-5

### Two paths

#### Path A: Use existing PRM (recommended to start)

Math-Shepherd is an open-source PRM trained on math reasoning.
Load it, wrap in a Ray actor, done.

```
Input:  "Step 1: 16 - 3 = 13. Step 2: 13 - 4 = 9. Step 3: 9 * 2 = 18"
Output: [1.0, 1.0, 1.0]  (score per step)

Input:  "Step 1: 16 - 3 = 12. Step 2: 12 - 4 = 8. Step 3: 8 * 2 = 16"
Output: [0.0, 0.0, 0.0]  (first step wrong, rest cascade)
```

#### Path B: Train your own PRM (later, if time permits)

1. Have the teacher (70B) solve 10K problems step by step
2. Programmatically verify each step (execute arithmetic)
3. Label: correct_step=1, incorrect_step=0
4. Train a 3B model on these labels

This is a separate mini-project. Do it AFTER the main pipeline works.

### Tasks

#### 6.1 — Load and test Math-Shepherd (Path A)

- Download from HuggingFace
- Test on 50 hand-verified solutions
- Verify it catches wrong steps correctly

#### 6.2 — Define scoring interface

```
RewardModel.score(solutions) → scores

Where scores can be:
    - Simple: one number per solution (sum of step scores)
    - Detailed: list of scores per step per solution

Start with simple. Use detailed later for analysis.
```

#### 6.3 — Handle the code-augmented case

If your student generates Python code to solve problems (rStar-style),
you have a simpler option: EXECUTE the code and check the output.

```
Student output:
    eggs = 16
    eaten = 3
    baked = 4
    remaining = eggs - eaten - baked
    revenue = remaining * 2
    print(revenue)  # 18

Execute in sandbox → output is 18 → compare to ground truth
This is a free, perfect reward signal for code solutions.
```

#### 6.4 — Integrate into Ray

Replace FakeRewardModel with real RewardActor.

### Checkpoint: Phase 6 is done when

- [ ] PRM loads and scores solutions correctly
- [ ] Tested on known-correct and known-incorrect solutions
- [ ] RewardActor integrates into Ray loop
- [ ] Scoring is fast enough to not bottleneck training

---

## Phase 7: GRPO Training with Teacher Intervention

**Goal:** The full system. Train the student with GRPO, teacher rescue,
and PRM scoring.
**Duration:** 2-3 weeks
**Cost:** ~$30-50

### This is where everything comes together

```
GPU 0                              GPU 1
┌────────────────────────┐         ┌──────────────────────────────┐
│                        │         │                              │
│  Teacher 70B           │         │  Student 3B                  │
│  (4-bit, SGLang)       │         │  (Unsloth + LoRA training)   │
│                        │         │                              │
│  Only activated when   │         │  Generates solutions         │
│  all solutions fail    │         │  Receives GRPO updates       │
│                        │         │                              │
│  ~35 GB VRAM           │         │  Reward Model (PRM)          │
│                        │         │  (inference, shared)         │
│                        │         │                              │
│                        │         │  ~10 GB + ~6 GB (with        │
│                        │         │   Unsloth memory savings)    │
└────────────────────────┘         └──────────────────────────────┘

         ↕ Ray manages all communication ↕

With Unsloth, the student GPU has plenty of headroom.
An A100 40GB or RTX 4090 (24GB) is sufficient for GPU 1.
Only GPU 0 (teacher) needs an A100 80GB.
```

### Tasks

#### 7.1 — Implement GRPO logic

The core algorithm:

```
Given solutions and rewards for one problem:

    mean_reward = average(rewards)

    for each solution_i:
        advantage_i = reward_i - mean_reward

        if advantage_i > 0:
            # This solution was better than average
            # Increase its probability
            loss += -advantage_i * log_prob(solution_i)
        else:
            # This solution was worse than average
            # Decrease its probability
            loss += -advantage_i * log_prob(solution_i)

    # The math works out the same — solutions with positive
    # advantage get reinforced, negative get discouraged.

    backpropagate(loss)
    update LoRA weights
```

You can use TRL's GRPOTrainer or implement this yourself for learning.
Implementing yourself = deeper understanding, more bugs to fix.
Using TRL = faster, battle-tested, less learning.

**Recommendation:** Use TRL's GRPOTrainer, but read and understand the
source code so you know what it's doing.

#### 7.2 — Implement teacher intervention logic

```
The conditional rescue:

    solutions = student.generate(problem, n=8)
    rewards = reward_model.score(solutions)

    if max(rewards) > CORRECT_THRESHOLD:
        # At least one solution is correct(ish)
        # Normal GRPO — student can learn from its own attempts
        grpo_update(solutions, rewards)
        teacher_intervened = False
    else:
        # All solutions failed
        # Teacher analyzes and gives hint
        hint = teacher.analyze_and_hint(problem, solutions)

        # Student retries with teacher guidance
        retry_solutions = student.generate_with_hint(problem, hint, n=8)
        retry_rewards = reward_model.score(retry_solutions)

        # GRPO on retries
        grpo_update(retry_solutions, retry_rewards)
        teacher_intervened = True

    log(teacher_intervened)
```

#### 7.3 — Set up logging and monitoring

Use Weights & Biases (wandb) to track in real time:

```
Metrics to log every N steps:
    - Training loss
    - Average reward per batch
    - Teacher intervention rate (% of batches needing teacher)
    - GSM8K accuracy (evaluate every 500-1000 steps on a subset)

What the dashboard should show over training:

    Accuracy:             45% ──────→ 55% ──────→ 65%
    Teacher intervention: 70% ──────→ 30% ──────→ 5%
    Average reward:       0.1 ──────→ 0.4 ──────→ 0.7
```

#### 7.4 — Checkpointing strategy

```
Save LoRA adapter every 500 steps → HuggingFace Hub
If Vast.ai instance dies:
    - Rent new instance
    - Load base model + latest LoRA checkpoint
    - Resume training from last step

    You lose at most 500 steps of work.
```

#### 7.5 — Run training

```
Training plan:
    Phase 7a: GRPO WITHOUT teacher (control experiment)
        - Train for ~5K steps
        - Evaluate → expect ~58-62%
        - Save model

    Phase 7b: GRPO WITH teacher (your innovation)
        - Same setup, same steps
        - Teacher intervenes when needed
        - Evaluate → expect ~63-67%
        - Save model

    The DIFFERENCE between 7a and 7b is your contribution.
    This is what makes the project novel.
```

#### 7.6 — Hyperparameter tuning

Things you may need to adjust:
- CORRECT_THRESHOLD: when does the teacher intervene?
  Too low = teacher always intervenes (student depends on it)
  Too high = teacher never intervenes (defeats the purpose)
  Start at 0.3, adjust based on intervention rate.

- Number of solutions (n): 8 is standard, 4 is cheaper, 16 is better but slow

- Hint strength: how much does the teacher give away?
  Experiment with light hints vs heavy hints.

- KL penalty: prevents the student from changing too fast.
  Too high = slow learning. Too low = unstable training.

### Checkpoint: Phase 7 is done when

- [ ] GRPO training runs without crashes for 5K+ steps
- [ ] Teacher intervention works correctly (activates only on failure)
- [ ] Metrics are logged to wandb
- [ ] Checkpoints save to HuggingFace Hub
- [ ] You have results for GRPO without teacher (control)
- [ ] You have results for GRPO with teacher (experiment)
- [ ] The teacher version outperforms the non-teacher version

---

## Phase 8: Evaluation, Ablations, and Presentation

**Goal:** Prove it works. Understand why. Present it well.
**Duration:** ~1 week
**Cost:** ~$5

### Tasks

#### 8.1 — Final evaluation

Run all models on the full GSM8K test set (1,319 problems):

```
Results table:

    Model                              GSM8K Accuracy
    ─────────────────────────────────  ──────────────
    Llama 3.2 3B (baseline)            ~45%
    + SFT on MetaMathQA                ~55-60%
    + GRPO (no teacher)                ~58-62%
    + GRPO with teacher (yours)        ~65%+
```

#### 8.2 — Ablation studies

These strengthen your results by showing each component matters:

```
Ablations:
    1. Teacher hint strength
       - No hint (baseline GRPO)
       - Light hint ("check your subtraction")
       - Medium hint ("subtract eggs eaten before calculating remainder")
       - Heavy hint (nearly full solution)
       → Which works best?

    2. Intervention threshold
       - Teacher at 0.1 (intervenes on almost everything)
       - Teacher at 0.3 (intervenes on clear failures)
       - Teacher at 0.5 (only intervenes on total failures)
       → What's the sweet spot?

    3. Number of GRPO samples
       - n=4 vs n=8 vs n=16
       → Diminishing returns?
```

#### 8.3 — Qualitative analysis

Pick 10 interesting examples and show:

```
Problem: "..."

Before training (baseline):
    "16 + 3 = 19... the answer is 42" (completely wrong)

After SFT:
    "16 - 3 = 13, 13 * 2 = 26" (right idea, wrong execution)

After GRPO (no teacher):
    "16 - 3 = 13, 13 - 4 = 9, 9 * 2 = 18" (correct!)

A hard problem where only teacher-guided GRPO succeeds:
    Student's failed attempts: [all wrong in the same way]
    Teacher hint: "Split the discount into per-item before totaling"
    Student retry: [correct solution using the hint]
```

#### 8.4 — Write blog post

Structure:
1. The problem (small models are bad at math)
2. The idea (teacher intervenes only when GRPO fails)
3. The system (architecture diagram)
4. Results (table + charts)
5. What surprised you (learnings)
6. What you'd do differently

#### 8.5 — Clean up the repository

```
Final repo structure:

math-reasoning-distill/
├── README.md               (project overview + results)
├── pyproject.toml
├── configs/
│   ├── train_sft.yaml
│   ├── train_grpo.yaml
│   ├── train_grpo_teacher.yaml
│   └── vast_setup.sh
├── src/
│   ├── data/
│   │   ├── load_gsm8k.py
│   │   ├── load_metamath.py
│   │   └── normalize.py
│   ├── models/
│   │   ├── teacher.py
│   │   ├── student.py
│   │   └── reward.py
│   ├── training/
│   │   ├── sft.py
│   │   ├── grpo.py
│   │   └── coordination.py
│   └── eval/
│       ├── gsm8k_eval.py
│       └── metrics.py
├── scripts/
│   ├── run_baseline.py
│   ├── run_sft.py
│   ├── run_grpo.py
│   └── run_grpo_teacher.py
├── results/
│   ├── baseline_metrics.json
│   ├── sft_metrics.json
│   ├── grpo_metrics.json
│   ├── grpo_teacher_metrics.json
│   └── ablations/
└── blog/
    └── post.md
```

### Checkpoint: Phase 8 is done when

- [ ] All models evaluated on full GSM8K test set
- [ ] Results table clearly shows improvement at each stage
- [ ] At least 2 ablation studies completed
- [ ] 10 qualitative examples prepared
- [ ] Blog post written
- [ ] Repository cleaned up and README written
- [ ] Trained model uploaded to HuggingFace Hub

---

## Complete Timeline and Budget

```
Phase   What                        Duration    Cost        Cumulative
─────   ────                        ────────    ────        ──────────
  0     Study foundations           1 week      $0          $0
  1     Data pipeline               1 week      $0          $0
  2     Baseline evaluation         3-5 days    $1-2        $1-2
  3     SFT training (Unsloth)      1-2 weeks   $2-5        $3-7
  4     Learn Ray (fake models)     1 week      $0          $3-7
  5     Teacher setup (70B)         1 week      $5-10       $8-17
  6     Reward model (PRM)          1 week      $2-4        $10-21
  7     GRPO + teacher (Unsloth)    2-3 weeks   $20-35      $30-56
  8     Evaluation + presentation   1 week      $3          $33-59
        ──────────────────────────────────────────────────────────
        TOTAL                       ~10 weeks   $33-59

(Unsloth cuts training costs ~40-50% by enabling cheaper GPUs
 and finishing faster on the same hardware.)
```

### Money-saving rules

1. NEVER write or debug code on a paid GPU instance
2. Develop locally, test on CPU with 5 samples, then run on GPU
3. Script your Vast.ai environment setup (vast_setup.sh)
4. Use bid/spot instances for experimentation (30-50% cheaper)
5. Use on-demand instances only for final training runs
6. Checkpoint every 500 steps to HuggingFace Hub
7. Shut down instances THE MOMENT you're done
8. Monitor your Vast.ai spending daily

### Git workflow

```
Every phase produces a tagged commit:

    git tag phase-1-data
    git tag phase-2-baseline
    git tag phase-3-sft
    ...

This lets you always go back to a known-good state.
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GRPO training is unstable | Start with SFT (Phase 3) — you have results even if GRPO fails |
| 70B teacher doesn't fit on GPU | Use 4-bit quantization (AWQ). Fallback: use 8B as teacher |
| Teacher hints are too vague or too specific | Iterate on prompt (Phase 5.3). This is a tunable parameter |
| Vast.ai instance dies mid-training | Checkpoint every 500 steps to HuggingFace Hub |
| GSM8K improvement is less than 20% | Even 10% improvement with clear methodology is publishable |
| Budget runs out | Each phase is independently valuable. Stop at any phase and you still have something to show |

---

## What This Project Teaches You

By the end, you will have hands-on experience with:

- Fine-tuning LLMs (SFT + LoRA via Unsloth)
- Reinforcement learning for LLMs (GRPO)
- Optimized training workflows (Unsloth — custom CUDA kernels, memory optimization)
- Multi-model orchestration (Ray)
- Model serving (SGLang/vLLM, quantization)
- Reward modeling (PRM)
- Distributed GPU computing (Vast.ai)
- Rigorous evaluation (benchmarking, ablations)
- The full ML experiment lifecycle (data → training → evaluation → presentation)

These are exactly the skills that AI labs and ML teams hire for.
