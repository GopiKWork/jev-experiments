# System 1 Decision Model Experiments

These examples compare System 1 decision models for a car dealership scenario. A System 1 model answers typed questions in one forward pass and returns calibrated probabilities instead of free text. Jev is the TypeSafe model, which you call through OpenRouter. You do not need a TypeSafe API key. Laya and Decider are open-weight models from Hugging Face, which run on your machine. The steering examples also use an OpenAI model on Amazon Bedrock as the agent.

| Example | What it shows |
|---|---|
| `jev_basics.py` | The three Jev question types in a single request |
| `laya_basics.py` | The same request, answered by Laya running locally |
| `decider_basics.py` | The same request, answered by Decider running locally |
| `steering_demo.py` | Jev, Laya, and Decider as ways to steer a Strands agent, next to deterministic code and an LLM |
| `benchmark.py` | Accuracy and latency of the five steering methods across 50 scenarios |
| `model_architecture.ipynb` | How Laya and Decider differ from a standard LLM, read from their weights |
| `trace_decider.py` | One Decider forward pass traced by hand and checked against the API |
| `running_example.py` | The same three questions about one message, on all three models |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) is installed and on your `PATH`. To install it, run the following command and then open a new shell.

  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- You have an OpenRouter API key from https://openrouter.ai/keys.
- For `steering_demo.py` and `benchmark.py` only: you have AWS credentials with access to the OpenAI GPT-5.6 Terra model (`us.openai.gpt-5.6-terra`) on Amazon Bedrock in `us-east-1`.
- For Laya only: about 1 GB of free disk space for the weights. Laya runs on CPU. A GPU is optional.
- For Decider only: about 9 GB of free disk space for the weights. Decider runs on CPU. A GPU is optional.
- For the notebook only: about 11 GB of free memory, because it loads both open-weight models. It releases Laya before loading Decider.

## Setup

1. Clone the repository and change into it.

   ```
   git clone https://github.com/GopiKWork/jev-experiments.git
   cd jev-experiments
   ```

2. Copy the example environment file and paste your OpenRouter API key into the copy. The `.env` file is ignored by git, so your key stays on your machine.

   ```
   cp .env.example .env
   ```

   ```
   OPENROUTER_API_KEY=sk-or-...
   ```

3. Create the virtual environment and install the dependencies.

   ```
   uv venv --python 3.12
   uv pip install -r requirements.txt
   ```

## Example 1: Jev basics

The script sends one customer message to Jev with three questions. It then uses plain Python to decide whether the vehicle gets a same-day priority service booking.

| Type | Returns | Example in script |
|---|---|---|
| Choice | One label from a set, with a confidence | Which department handles the message: Service, Parts, Sales, or Warranty Claims |
| Score | A float on an ordered rubric | How soon the vehicle needs attention (0 to 2) |
| Noul | Probability that the answer is yes (0 to 1) | Whether it is unsafe to keep driving the vehicle |

The script books a same-day priority slot when `safety_risk` is 0.6 or higher, or when `urgency` is 1.5 or higher.

```
uv run python jev_basics.py
```

The output looks like this. The exact values vary.

```
department:  Service (1.00)
urgency:     Immediately (2.00)
safety_risk: 0.89
action:      same-day priority booking (Service)
```

## Example 2: Laya basics

[Laya](https://huggingface.co/convaiinnovations/laya) is an Apache 2.0 open-weight System 1 decision model. It has a ModernBERT-large backbone and 421M parameters. It accepts the same Choice, Score, and Noul questions as Jev, and it answers all questions in one forward pass. The `laya` Python package loads the weights from Hugging Face and runs them locally.

`laya_basics.py` sends the same message and questions as Example 1. Questions are plain dicts instead of SDK classes.

```
uv run python laya_basics.py
```

On the first run, `laya.load` calls `huggingface_hub.snapshot_download` and writes about 800 MB of weights to the Hugging Face cache, not to this directory. The default location is `~/.cache/huggingface/hub/models--convaiinnovations--laya`. Set `HF_HOME` to move the cache elsewhere. After the first run, the model loads in about 5 seconds. Each prediction takes about 0.5 seconds on an 8-core CPU. The output from a test run follows. Laya is deterministic, so the values repeat.

```
department:  Warranty Claims (0.86)
urgency:     Within a week (1.46)
safety_risk: 0.66
action:      same-day priority booking (Warranty Claims)
```

Laya reaches the same same-day decision as Jev, but routes the message to Warranty Claims because the customer mentions the warranty. Jev routes it to Service.

## Example 3: Decider basics

[Decider](https://github.com/Mapika/decider) is an Apache 2.0 open-weight System 1 decision model and an independent reproduction of the model class. It is not affiliated with TypeSafe. The `decider-4b` checkpoint has 4.2B parameters and is fine-tuned from Qwen3.5-4B-Base. It answers the same Choice, Score, and Noul questions as Jev and Laya.

`decider_basics.py` sends the same message and questions as Examples 1 and 2.

```
uv run python decider_basics.py
```

The script pins the `v2` revision. The `main` branch carries v2.1, which the model card describes as less well calibrated on hard items than v2. On the first run, `snapshot_download` writes about 8.4 GB of weights to `~/.cache/huggingface/hub/models--Mapika--decider-4b`. The output from a test run follows.

```
department:  Service (0.81)
urgency:     Immediately (1.94)
safety_risk: 0.98
action:      same-day priority booking (Service)
```

### Running on CPU

Decider-4b is a Qwen3.5 hybrid model. Of its 32 layers, 24 use linear attention and 8 use full attention, in a repeating pattern of three linear layers followed by one full layer. When the `flash-linear-attention` package is importable, transformers routes the linear-attention layers through a Triton kernel that requires a GPU driver, and the script fails with `RuntimeError: 0 active drivers`. Installing `decider-ai` pulls that package in as a dependency, so the failure occurs on any machine without a GPU.

The script therefore hides the package before importing transformers, which selects the reference PyTorch implementation instead.

```python
sys.modules["fla"] = None
```

The reference path is correct but slower. Delete the line when you run on a GPU. A full run of this script takes about 11 seconds on an 8-core CPU, including the 5 seconds to load the weights.

### The three models on one message

All three models answer the same three questions about the same customer message.

| Model | Runs | department | urgency | safety_risk |
|---|---|---|---|---|
| Jev | OpenRouter | Service (1.00) | Immediately (2.00) | 0.88 |
| Laya | Locally, 421M | Warranty Claims (0.86) | Within a week (1.46) | 0.66 |
| Decider | Locally, 4.2B | Service (0.81) | Immediately (1.94) | 0.98 |

Decider agrees with Jev on the department and on the urgency level, and it is the most confident of the three that the vehicle is unsafe to drive. Laya is the only model that routes the message to Warranty Claims.

Note that `decider-ai` pins `numpy` below 2.0. Installing it downgrades numpy from 2.x to 1.26.4. Examples 1 and 2 were rerun after the downgrade and produce unchanged results.

## Example 4: Steering a Strands agent

[Strands steering](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/steering/) runs a handler before the agent calls a tool. The handler returns `Proceed` to let the call run, or `Guide` to cancel the call and send feedback to the agent. The agent then retries with the feedback.

### Scenario

A dealership agent runs on GPT-5.6 Terra through Amazon Bedrock and books appointments with the `book_service` tool. Its system prompt tells it to always use standard priority unless a tool result says otherwise. The customer writes: "When I stop at lights the car pulls hard to the left and the pedal sinks almost to the floor." This is a brake failure, but the message never uses the word "brake".

All five methods enforce the same dealership policy:

- Route the booking to one department: Service, Parts, Sales, or Warranty Claims.
- Use priority `same-day` only for a Service booking where the vehicle is unsafe to drive. Use `standard` for everything else.

### Five steering methods

| Method | Who decides | Strength | Weakness |
|---|---|---|---|
| `deterministic` | Python keyword lists for safety and department | Fast, cheap, and predictable | Misses anything the keyword list does not cover |
| `llm` | A second LLM that reads the policy (`LLMSteeringHandler`) | Understands language | Slower, costs more, and the decision is free text |
| `jev` | Jev classifies, then Python `if` statements decide | Understands language and returns probabilities that code can threshold | Needs well-scoped questions and a network call |
| `laya` | Laya classifies locally, then the same Python `if` statements decide | Fastest local option, no API cost, and results are deterministic | Least accurate of the three model-based methods |
| `decider` | Decider classifies locally, then the same Python `if` statements decide | Close to Jev on accuracy with no API cost | About six times slower per call than Laya, and 8.4 GB of weights |

Both the agent and the `llm` steering handler use GPT-5.6 Terra. The `jev`, `laya`, and `decider` methods ask the same two questions: `safety_risk` (Noul) and `department` (Choice). Plain Python compares the answers against the booking and returns `Guide` or `Proceed`. The LLM agent takes the action. The policy stays in code that you can read and test.

Swapping one classifier for another changes only the class that answers the questions. The policy in `check` is shared by all three.

### Run

```
uv run python steering_demo.py deterministic
uv run python steering_demo.py llm
uv run python steering_demo.py jev
uv run python steering_demo.py laya
uv run python steering_demo.py decider
```

The output from a test run follows. The exact wording varies.

```
[method] deterministic
[steer] Proceed: booking matches policy
[tool]  booked Service / standard
```

```
[method] llm
[steer] Guide: The notes describe a safety issue, so the booking priority must be changed from "standard" to "same-
[steer] Proceed: The prior book_service attempt was cancelled because the safety-related notes used priority "standar
[tool]  booked Service / same-day
```

```
[method] jev
[jev]   safety_risk=0.94 department=Service
[steer] Guide: Book department 'Service' with priority 'same-day'.
[jev]   safety_risk=0.95 department=Service
[steer] Proceed: booking matches policy
[tool]  booked Service / same-day
```

```
[method] laya
[laya]  safety_risk=0.49 department=Service
[steer] Proceed: booking matches policy
[tool]  booked Service / standard
```

```
[method] decider
[decider] safety_risk=0.96 department=Service
[steer] Guide: Book department 'Service' with priority 'same-day'.
[decider] safety_risk=0.96 department=Service
[steer] Proceed: booking matches policy
[tool]  booked Service / same-day
```

The LLM, Jev, and Decider methods catch the brake failure and guide the agent to a same-day booking. The deterministic method misses it because no keyword matches. Laya routes the booking correctly, but scores the safety risk at 0.49, below the 0.6 threshold. Each run ends with a one-sentence `[agent]` reply to the customer.

## Example 5: Benchmark the steering methods

`benchmark.py` runs every scenario in `scenarios.jsonl` through the agent once per steering method. It then reports accuracy and latency for each method.

### Scenarios

`scenarios.jsonl` holds 50 customer messages, one JSON object per line. Each object stores the ground truth: the booking that correct steering should produce.

```
{"id": 21, "message": "The brake light bulb on the rear passenger side is out. Can you replace it when convenient?", "expected_department": "Service", "expected_priority": "standard", "rationale": "Minor bulb replacement. The word brake is a trap."}
```

| Group | Count | Expected booking |
|---|---|---|
| Unsafe vehicle | 19 | Service, same-day |
| Routine or cosmetic repair | 14 | Service, standard |
| Buy, lease, or trade in | 6 | Sales, standard |
| Buy parts only | 6 | Parts, standard |
| Refund or warranty dispute | 5 | Warranty Claims, standard |

The scenarios are subjective by design. Many unsafe cases never use a safety keyword, such as "the pedal sinks almost to the floor". Many safe cases do use one, such as a peeling steering wheel, a stale smoke smell, or brake pads a customer wants to buy and fit themselves. The `rationale` field explains each label. To add scenarios, append lines to the file.

### Metrics

| Metric | Definition |
|---|---|
| accuracy | Share of scenarios where the final booking matches both `expected_department` and `expected_priority` |
| total (s) | Wall-clock time for the method to finish all scenarios |
| avg (s) | Mean end-to-end latency per scenario, from the customer message to the agent's final reply. This includes agent calls, steering calls, and retries after `Guide`. |
| errors | Scenarios where the run raised an exception. These count as incorrect. |
| missed ids | Scenario IDs with a wrong or missing booking |

### Run

```
uv run python benchmark.py                   # all five methods, all scenarios
uv run python benchmark.py jev laya          # selected methods
uv run python benchmark.py --limit 10        # first 10 scenarios only
uv run python benchmark.py --workers 1       # one scenario at a time
```

By default, 8 scenarios run in parallel. A full run of all five methods takes about 4 minutes, most of it Decider. The benchmark loads Laya and Decider before timing starts, so load time is not counted.

### Results

The output of a full run on 2026-09-25 with 8 workers, on an 8-core CPU with no GPU, follows.

```
method          accuracy   total (s)   avg (s)  errors  missed ids
deterministic        56%        10.9       1.6       0  [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 19, 21, 22, 23, 24, 28, 42, 48]
llm                  80%        19.9       3.0       0  [5, 6, 7, 8, 9, 11, 14, 16, 17, 18]
jev                  98%        14.5       2.1       0  [21]
laya                 72%        35.0       5.2       0  [1, 2, 5, 7, 8, 10, 11, 14, 15, 16, 17, 36, 37, 48]
decider              94%       174.5      26.5       0  [7, 11, 28]
```

- **Deterministic** is the fastest method but the least accurate. It misses unsafe cases that do not contain a keyword, and it flags safe cases that do (21 to 24 and 28). Its keyword lists also route some Parts and Warranty Claims messages to the wrong department (42 and 48).
- **LLM** fixes most routing, but it often lets an unsafe vehicle through at standard priority. The set of missed IDs changes between runs. Each steering decision is a second GPT-5.6 Terra call, so its latency is high.
- **Jev** is the most accurate. Its only miss is scenario 21, where it scored a burnt-out brake light bulb at 0.67 safety risk, just above the 0.6 threshold.
- **Laya** gets no false alarms on the keyword traps. Most of its misses are unsafe vehicles that it scores below the 0.6 threshold (1 to 17). It also routes two Sales messages (36 and 37) and one Warranty Claims message (48) to the wrong department. Its accuracy and missed IDs were identical across runs.
- **Decider** comes within 4 points of Jev with no network call and no API key. It misses only three scenarios, and it is the only local method that catches the keyword-free hazards that Laya scores too low. It is also the slowest by a wide margin.

The two local methods have inflated average latencies for the same reason. One model instance serves all 8 parallel scenarios, and a lock lets only one prediction run at a time, so most of the average is queueing. Measured alone, a Laya prediction takes about 0.5 seconds and a Decider prediction about 3 seconds on an 8-core CPU. With `--workers 1`, Laya averages 2.2 seconds per scenario and Jev averages 2.0 seconds. Decider would be faster on a GPU, where it can use the Triton kernel described in Example 3 instead of the reference PyTorch path.

### Accuracy against cost

| Method | Accuracy | Needs a network call | Weights on disk |
|---|---|---|---|
| `deterministic` | 56% | No | None |
| `laya` | 72% | No | 0.8 GB |
| `llm` | 80% | Yes | None |
| `decider` | 94% | No | 8.4 GB |
| `jev` | 98% | Yes | None |

The ranking is the useful result. A 421M-parameter local model beats hand-written keyword rules by 16 points. A 4.2B local model beats a frontier LLM used as a judge by 14 points, at a tenth of Jev's speed but with no data leaving the machine. Jev remains the most accurate.

The Laya model card notes that the shipped checkpoint is overconfident and recommends fitting a calibration temperature on your own data. A lower threshold for `LayaSteering.THRESHOLD` may improve its recall on unsafe vehicles.

## Example 6: Model architecture notebook

`model_architecture.ipynb` opens Laya and Decider side by side with a conventional LLM and locates the exact point where the architectures diverge. It is written for readers who want to see the tensors rather than take the claim on trust.

```
uv run jupyter notebook model_architecture.ipynb
```

The notebook is ordered so the cheap work comes first. Sections 1 to 5 read `config.json` files and safetensors headers, which needs no GPU and almost no memory, because a safetensors header lists every tensor name, shape, and dtype without allocating a parameter. Sections 6 and 7 load the weights and run instrumented forward passes. The gpt-oss baseline column costs a few kilobytes, because only its `config.json` is downloaded.

| Section | What it establishes |
|---|---|
| 1. Setup | A safetensors header reader, used by the next four sections |
| 2. Three architectures at a glance | Laya is an encoder, Decider is a linear-attention hybrid, gpt-oss is a mixture of experts |
| 3. Weight inventory | Laya spends 6.3% of its parameters on decision machinery. Decider spends 0%. |
| 4. Where the distribution comes from | Both decision heads, quoted from the installed packages |
| 5. Attention | Bidirectional, causal, and sliding masks printed next to each other |
| 6. One instrumented forward pass | Real tensor shapes from both models on the same customer message |
| 7. Calibration temperature | The fitted per-question-type temperatures, and their effect on one answer |

The central finding is that the two models reach the same output shape by opposite routes. Laya replaces the language-model head with a trained scorer that maps the hidden state at each option marker to a single number. Decider keeps the Qwen3.5 head and slices it, using 255 of its 248,320 vocabulary rows, which is 0.103 percent, and adds no parameters at all. A conventional LLM uses the whole matrix and then runs the model again for every token it emits.

`build_notebook.py` generates the notebook. Editing cells in a Python file keeps them reviewable in a diff. To regenerate and re-run:

```
uv run python build_notebook.py
uv run python -m nbconvert --to notebook --execute --inplace model_architecture.ipynb
```

Results vary with model versions and network conditions. Rerun the benchmark before you rely on these numbers.

## Example 7: Trace a Decider forward pass by hand

`trace_decider.py` rebuilds the prompt rows that `system_one` builds, runs the backbone once, reads the answer slot, slices the output head to the option letters, applies the calibration temperature, and prints each hand-computed value next to the one the public API returns. Use it to confirm that the documented read path is the real one.

```
uv run python trace_decider.py           # bare option labels
uv run python trace_decider.py described # the criteria descriptions steering_demo.py sends
```

The two question sets matter. With bare labels the Choice prompt is 58 tokens; with descriptions it is 112 and the answer changes. Quote whichever one you ran.

| Section | What it prints |
|---|---|
| 1. Prompt | Token count, answer slot, and the last four token ids |
| 2. Head | The tied head, the 255-row slice, and the letter ids |
| 3. Choice | Raw logits, masked logits, and softmax at T=1 and at the config T |
| 4. API | The same question through `system_one`, key for key |
| 5. Vocabulary | The same hidden state against all 248,320 rows, unmasked |
| 6. Padding | The slot with and without the pad to a multiple of 64 |
| 7. Noul | The two-option row a Noul answer is read from |
| 8. Score | Isolated levels, `fit_mass`, and the listwise readout for comparison |

Two results are worth knowing. The head is tied to the input embeddings, confirmed by `data_ptr()` equality, and no `lm_head` tensor appears in the safetensors at all. And padding is not quite inert in bf16: the same row scores 0.7948 alone and 0.7946 padded to 64. The script prints both, and every published figure uses the padded path, because that is what `system_one` runs.

## Example 8: One message, three models

`running_example.py` asks the same three questions about one customer message on each model, so the answers can be compared directly. The message contains no safety keyword, which is why a keyword rule misses it.

```
uv run python running_example.py          # all three
uv run python running_example.py decider  # one
```

| Model | Department, and its probability | Urgency | Safety risk |
|---|---|---|---|
| `jev` | Service 1.00 | 2.00 Immediately | 0.94 |
| `laya` | Service 0.4123 | 0.9312 Within a week | 0.4949 |
| `decider` | Service 0.7946 | 1.65 Immediately | 0.9589 |

The table quotes the winning label's probability, because that is the one quantity all three report the same way. Their `confidence` fields are not comparable: Jev returns 1.00 here, Decider returns 0.7261 rather than its 0.7946 probability, and Laya returns both a `confidence` that is a normalised entropy and an `answer_confidence` that is `max(p)`, the second being the one its card calls calibrated. Threshold on a field you have checked the definition of.

Laya scores this message as a coin flip on safety and places urgency nearer "Within a week" for a vehicle whose brake pedal sinks to the floor. It runs here as shipped, with no calibration refit, which its model card advises against. Note that temperature scaling cannot fix that particular answer: it is monotone, so no temperature moves 0.4949 across 0.5. The card's own suggestion for Noul questions that track their label text is to recast them as a two-option Choice.

Jev is hosted and not bit-reproducible. Repeated calls on this message returned safety risk 0.94 and 0.95.

## Files

| File | Purpose |
|---|---|
| `jev_basics.py` | Example 1 |
| `laya_basics.py` | Example 2 |
| `decider_basics.py` | Example 3 |
| `steering_demo.py` | Example 4 |
| `benchmark.py` | Example 5 |
| `scenarios.jsonl` | 50 test scenarios with ground truth for Example 5 |
| `model_architecture.ipynb` | Example 6 |
| `build_notebook.py` | Generates `model_architecture.ipynb` |
| `trace_decider.py` | Example 7 |
| `running_example.py` | Example 8 |
| `requirements.txt` | Python dependencies |
| `.env` | Your OpenRouter API key. Git ignores this file. |
| `.env.example` | Template for `.env` |

## Configuration

| Setting | Value |
|---|---|
| Jev model ID | `~typesafe/jev-latest` |
| Jev base URL | `https://openrouter.ai/api` |
| Laya model | `convaiinnovations/laya` (English root checkpoint, run locally) |
| Decider model | `Mapika/decider-4b`, revision `v2` (run locally) |
| Agent model ID | `us.openai.gpt-5.6-terra` (OpenAI GPT-5.6 Terra on Amazon Bedrock) |
| Agent AWS Region | `us-east-1` |
| Agent `max_tokens` | 1024 |
| Safety threshold | 0.6 (`JevSteering.THRESHOLD`, `LayaSteering.THRESHOLD`, `DeciderSteering.THRESHOLD`) |
| Benchmark parallel workers | 8 (`--workers`) |

## References

- TypeSafe quick start: https://docs.typesafe.ai/introduction/quickstart
- TypeSafe use case map: https://docs.typesafe.ai/concepts/use-case-map
- Laya model card: https://huggingface.co/convaiinnovations/laya
- Decider source: https://github.com/Mapika/decider
- Decider model card: https://huggingface.co/Mapika/decider-4b
- OpenAI models on Amazon Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html
- Strands steering: https://strandsagents.com/docs/user-guide/sdk/agents/interventions/steering/
- Source notebook: https://github.com/mikegc-aws/jev-strands-video/blob/main/demos/jev_basics/jev_basics.ipynb
