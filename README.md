# 🐝 SwarmGPT

> Parallel LLM swarm intelligence — run N models simultaneously across Ollama, OpenAI, and Anthropic, then use Particle Swarm Optimization to converge on the best possible answer.

I was inspired after reading The Art Of Randomness by Ronald T. Kneusel, particually around Swarm based Algorithms and playing around with various Nature Based Swarm Optimization Algorithms. so this is a thought exercise of can we give small inference to mini agents and collectivly swarm to a final answer.

---

## How It Works

SwarmGPT treats each LLM call as a **particle in a swarm**. Every agent gets a unique temperature, fires inference in parallel, and its response is scored by a fitness function. The swarm then collectively evolves toward better answers across multiple iterations — inspired by [Particle Swarm Optimization (PSO)](https://en.wikipedia.org/wiki/Particle_swarm_optimization).

```
Iteration 1 ──► 20 agents fire in parallel
                  ↓
             Score each response (fitness function)
                  ↓
             Synthesize top-5 into aggregate answer
                  ↓
             PSO: nudge each agent's temperature
                  toward the best-performing region
                  ↓
Iteration 2 ──► 20 agents fire again (smarter temps)
                  ↓
                 ...
                  ↓
Iteration N ──► Final synthesized answer
```

### The Fitness Function

Each response is scored across four dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Keyword coverage | 35% | How well the response addresses the original prompt |
| Length adequacy | 25% | Sweet spot around 150 words — not too short, not bloated |
| Coherence | 25% | Unique vs repeated sentences — penalizes rambling |
| Diversity bonus | 15% | Rewards agents that surface unique insights vs the pack |

### PSO Temperature Update

Each agent's `temperature` is its **position** in the swarm. After every iteration, velocities are updated using the classic PSO rule:

```
velocity = inertia × velocity
         + cognitive × r1 × (personal_best_score - current_score)
         + social    × r2 × (global_best_temp - current_temp)
```

Agents drift toward temperatures that historically produced high-scoring responses, while maintaining enough diversity to keep exploring.

---

## Providers

| Provider | Concurrency | Notes |
|---|---|---|
| **Ollama** | Unlimited | Local, free, no rate limits |
| **OpenAI** | Semaphore-capped (default: 20) | Safe for Tier 1 (500 RPM) |
| **Anthropic** | Semaphore-capped (default: 15) | Safe for Tier 1 (50 RPM) |

Cloud providers use `threading.Semaphore` so agents queue gracefully — you'll never get a 429 from firing 20 simultaneous requests.

---

## Installation

**1. Clone and install dependencies**

```bash
git clone https://github.com/yourname/swarmgpt
cd swarmgpt
pip install requests rich numpy python-dotenv openai anthropic
```

**2. Set up your `.env`**

```bash
cp .env.example .env
```

Then edit `.env` with your keys and preferred models (see [Configuration](#configuration) below).

**3. Make sure Ollama is running** *(if using Ollama)*

```bash
ollama serve
# SwarmGPT will auto-pull the model if it's not already downloaded
```

---

## Quick Start

```bash
# Ollama only (no API keys needed)
python swarmgpt.py --prompt "What Should I have for Lunch?"

# OpenAI, 20 agents, 5 iterations
python swarmgpt.py --prompt "What causes inflation?" --providers openai

# Mixed swarm across all three providers
python swarmgpt.py --prompt "Explain backpropagation" \
  --providers ollama openai anthropic \
  --agents 20 \
  --weights "ollama:10,openai:5,anthropic:5"

# Save output to a file
python swarmgpt.py --prompt "Write a product spec for a todo app" \
  --providers openai --agents 15 --iterations 4 --output result.txt
```

---

## Configuration

All defaults live in `.env`. CLI flags override them at runtime.

```env
# ── Ollama ──────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=tinyllama

# ── OpenAI ──────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_CONCURRENT=20

# ── Anthropic ───────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_MAX_CONCURRENT=15

# ── Swarm Defaults ───────────────────────────
SWARM_AGENTS=20
SWARM_ITERATIONS=5
SWARM_MAX_TOKENS=300
SWARM_TEMP_MIN=0.3
SWARM_TEMP_MAX=1.2

# ── Mixed Swarm Weights ──────────────────────
# Proportional allocation across providers
SWARM_PROVIDER_WEIGHTS=ollama:10,openai:5,anthropic:5
```

---

## CLI Reference

```
python swarmgpt.py [OPTIONS]

Required:
  --prompt,   -p    The question or task for the swarm

Providers:
  --providers       Space-separated list: ollama openai anthropic
                    (default: ollama)

Swarm:
  --agents,   -n    Total number of parallel agents (default: 20)
  --iterations, -i  PSO iterations (default: 5)
  --max-tokens      Max tokens per agent response (default: 300)
  --temp-min        Min agent temperature (default: 0.3)
  --temp-max        Max agent temperature (default: 1.2)
  --weights         Provider allocation e.g. "ollama:10,openai:5"

Output:
  --output,   -o    Save final answer to a file
```

---

## Example Output

```
🐝 SwarmGPT
Providers: ollama | openai  |  Agents: 20  |  Iterations: 5
Explain backpropagation in simple terms

  Agent allocation → ollama: 12 | openai: 8

⟳  Iteration 1/5
┌──────────────────────────────────────────────────────────────┐
│ Agent Scores — Iteration 1                                   │
├────┬──────────┬──────┬───────┬────────┬─────────────────────┤
│ ID │ Provider │ Temp │ Score │ Tokens │ Preview             │
├────┼──────────┼──────┼───────┼────────┼─────────────────────┤
│  3 │ openai   │ 0.71 │ 0.821 │   187  │ Backprop is how ... │
│  1 │ ollama   │ 0.54 │ 0.764 │   203  │ Think of it like... │
│ ...│          │      │       │        │                     │
└────┴──────────┴──────┴───────┴────────┴─────────────────────┘
  ✓ Iter 1 | Best: 0.821 (openai) | Global best: 0.821 | Tokens: 3842

...

🏆  SwarmGPT — Final Synthesized Answer
╔══════════════════════════════════════════════════════════════╗
║ Backpropagation is the process by which a neural network    ║
║ learns from its mistakes. After making a prediction, the    ║
║ network compares its output to the correct answer and       ║
║ calculates the error. It then works backwards through each  ║
║ layer — adjusting the weights of connections proportionally ║
║ to how much each one contributed to the error...            ║
╚══════════════════════════════════════════════════════════════╝
```
---

## Project Structure

```
swarmgpt/
├── swarmgpt.py      # Main script all swarm logic
├── .env.example     # Config template copy to .env
├── .env             # Your local config (gitignored)
└── README.md
```

---

## Roadmap

- [ ] Async I/O (`asyncio` + `aiohttp`) for even lower latency
- [ ] Web UI dashboard showing live swarm convergence
- [ ] Pluggable fitness functions (task-specific scoring)
- [ ] Export full swarm history to JSON
- [ ] Support for local HuggingFace models via `transformers`
- [ ] Multi-prompt tournament mode

---

## License

MIT
