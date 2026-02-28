"""
SwarmGPT Multi-Provider Parallel Swarm Intelligence
======================================================
Runs N agents across Ollama / OpenAI / Anthropic in parallel,
scores responses with a PSO-inspired fitness function, and
iteratively synthesizes a collective answer.

Providers:
  - Ollama   : fully local, no rate limits
  - OpenAI   : cloud, semaphore-throttled (safe for Tier 1)
  - Anthropic: cloud, semaphore-throttled (safe for Tier 1)

Usage:
  python swarmgpt.py --prompt "Explain quantum entanglement simply"
  python swarmgpt.py --prompt "..." --providers ollama openai --agents 20
  python swarmgpt.py --prompt "..." --providers anthropic --agents 10 --iterations 3

Requirements:
  pip install requests rich numpy python-dotenv openai anthropic
"""

import argparse
import json
import os
import re
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Load .env from cwd or script directory
load_dotenv(Path(__file__).parent / ".env")
load_dotenv()  # also load from cwd

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Provider Clients
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """Local Ollama inference no rate limits, unlimited concurrency."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = "ollama"

    def infer(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        start = time.time()
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return {
            "text": data.get("response", "").strip(),
            "tokens": data.get("eval_count", 0),
            "latency": round(time.time() - start, 2),
            "provider": self.provider,
            "model": self.model,
        }

    def check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                console.print(f"[yellow]Ollama: pulling '{self.model}'...[/yellow]")
                self._pull()
            return True
        except Exception as e:
            console.print(f"[red]Ollama unavailable at {self.base_url}: {e}[/red]")
            return False

    def _pull(self):
        with requests.post(
            f"{self.base_url}/api/pull", json={"name": self.model}, stream=True, timeout=300
        ) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        console.print(f"  [dim]{data['status']}[/dim]")


class OpenAIClient:
    """OpenAI inference with semaphore-based concurrency control."""

    def __init__(self, api_key: str, model: str, max_concurrent: int = 20):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        self.model = model
        self.provider = "openai"
        self._sem = threading.Semaphore(max_concurrent)

    def infer(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        with self._sem:
            start = time.time()
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=min(temperature, 2.0),
                max_tokens=max_tokens,
            )
            return {
                "text": resp.choices[0].message.content.strip(),
                "tokens": resp.usage.total_tokens,
                "latency": round(time.time() - start, 2),
                "provider": self.provider,
                "model": self.model,
            }

    def check(self) -> bool:
        try:
            self._client.models.retrieve(self.model)
            return True
        except Exception as e:
            console.print(f"[red]OpenAI check failed: {e}[/red]")
            return False


class AnthropicClient:
    """Anthropic inference with semaphore-based concurrency control."""

    def __init__(self, api_key: str, model: str, max_concurrent: int = 15):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.model = model
        self.provider = "anthropic"
        self._sem = threading.Semaphore(max_concurrent)

    def infer(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        with self._sem:
            start = time.time()
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=min(temperature, 1.0),  # Anthropic caps at 1.0
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "text": resp.content[0].text.strip(),
                "tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                "latency": round(time.time() - start, 2),
                "provider": self.provider,
                "model": self.model,
            }

    def check(self) -> bool:
        try:
            # Lightweight check just verify key is set
            return bool(self._client.api_key)
        except Exception as e:
            console.print(f"[red]Anthropic check failed: {e}[/red]")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Config & Agent
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_COLORS = {
    "ollama": "cyan",
    "openai": "green",
    "anthropic": "magenta",
}

@dataclass
class SwarmConfig:
    providers: list[str]                         # ["ollama", "openai", "anthropic"]
    num_agents: int = 20
    iterations: int = 5
    temperature_range: tuple = (0.3, 1.2)
    max_tokens: int = 300
    provider_weights: dict = field(default_factory=dict)  # {"ollama": 10, "openai": 5, ...}
    # PSO hyperparams
    pso_inertia: float = 0.5
    pso_cognitive: float = 1.5
    pso_social: float = 2.0


@dataclass
class Agent:
    id: int
    provider: str
    client: object          # one of the client classes above
    temperature: float
    response: str = ""
    score: float = 0.0
    best_response: str = ""
    best_score: float = 0.0
    velocity: float = 0.0
    tokens_used: int = 0
    latency: float = 0.0
    error: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Fitness / Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_response(response: str, prompt: str, all_responses: list[str]) -> float:
    if not response or len(response) < 20:
        return 0.0

    score = 0.0
    words = response.split()

    # Length sweet spot (100–250 words)
    length_score = min(len(words) / 150, 1.0) * 0.25
    score += length_score

    # Keyword coverage from prompt
    keywords = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
    hits = sum(1 for kw in keywords if kw in response.lower())
    score += (hits / max(len(keywords), 1)) * 0.35

    # Coherence: unique vs repeated sentences
    sentences = [s.strip().lower() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 10]
    if sentences:
        score += (len(set(sentences)) / len(sentences)) * 0.25

    # Diversity vs other agents (Jaccard dissimilarity)
    unique_words = set(words)
    others = [r for r in all_responses if r != response]
    if others:
        overlaps = []
        for other in others:
            other_words = set(other.split())
            union = unique_words | other_words
            if union:
                overlaps.append(len(unique_words & other_words) / len(union))
        score += (1.0 - (sum(overlaps) / len(overlaps))) * 0.15

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# PSO Update
# ─────────────────────────────────────────────────────────────────────────────

def pso_update_temperature(agent: Agent, global_best_temp: float, config: SwarmConfig) -> float:
    r1, r2 = random.random(), random.random()
    new_vel = (
        config.pso_inertia * agent.velocity
        + config.pso_cognitive * r1 * (agent.best_score - agent.score)
        + config.pso_social * r2 * (global_best_temp - agent.temperature)
    )
    new_temp = agent.temperature + new_vel
    lo, hi = config.temperature_range
    agent.velocity = new_vel
    return round(max(lo, min(hi, new_temp)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation (synthesis step)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_responses(
    top_agents: list[Agent],
    prompt: str,
    synth_client,  # whichever client is nominated for synthesis
    max_tokens: int,
) -> str:
    top_responses = [a for a in top_agents if a.response and not a.error][:5]
    if not top_responses:
        return ""

    numbered = "\n\n".join(
        f"[Agent {i+1} | {a.provider} | temp={a.temperature}]:\n{a.response}"
        for i, a in enumerate(top_responses)
    )

    synthesis_prompt = (
        f"You are a synthesis engine. Below are {len(top_responses)} AI responses to the same question.\n"
        f"Combine their best ideas into one coherent, high-quality answer. Eliminate redundancy.\n\n"
        f"ORIGINAL QUESTION:\n{prompt}\n\n"
        f"AGENT RESPONSES:\n{numbered}\n\n"
        f"SYNTHESIZED ANSWER:"
    )

    try:
        result = synth_client.infer(synthesis_prompt, temperature=0.4, max_tokens=max_tokens * 2)
        return result["text"]
    except Exception as e:
        console.print(f"[red]Synthesis failed: {e}[/red]")
        return top_responses[0].response if top_responses else ""


# ─────────────────────────────────────────────────────────────────────────────
# SwarmGPT Core
# ─────────────────────────────────────────────────────────────────────────────

class SwarmGPT:
    def __init__(self, config: SwarmConfig, clients: dict):
        self.config = config
        self.clients = clients           # {"ollama": OllamaClient, ...}
        self.agents: list[Agent] = []
        self.global_best_response: str = ""
        self.global_best_score: float = 0.0
        self.global_best_temp: float = 0.7
        self.history: list[dict] = []

    def _build_agents(self):
        """Distribute agents across providers according to weights."""
        weights = self.config.provider_weights
        active = [p for p in self.config.providers if p in self.clients]

        if not weights:
            # Equal split
            weights = {p: 1 for p in active}

        total_weight = sum(weights.get(p, 0) for p in active)
        allocations = {}
        allocated = 0
        for i, p in enumerate(active):
            if i == len(active) - 1:
                allocations[p] = self.config.num_agents - allocated
            else:
                allocations[p] = round((weights.get(p, 0) / total_weight) * self.config.num_agents)
                allocated += allocations[p]

        lo, hi = self.config.temperature_range
        all_temps = list(np.linspace(lo, hi, self.config.num_agents))
        random.shuffle(all_temps)

        self.agents = []
        agent_id = 0
        temp_idx = 0
        for provider, count in allocations.items():
            client = self.clients[provider]
            for _ in range(count):
                self.agents.append(Agent(
                    id=agent_id,
                    provider=provider,
                    client=client,
                    temperature=round(all_temps[temp_idx], 3),
                    velocity=random.uniform(-0.1, 0.1),
                ))
                agent_id += 1
                temp_idx += 1

        # Print allocation summary
        summary = " | ".join(
            f"[{PROVIDER_COLORS.get(p, 'white')}]{p}: {n}[/{PROVIDER_COLORS.get(p, 'white')}]"
            for p, n in allocations.items() if n > 0
        )
        console.print(f"  Agent allocation → {summary}")

    def _run_agent(self, agent: Agent, prompt: str) -> Agent:
        try:
            result = agent.client.infer(prompt, agent.temperature, self.config.max_tokens)
            agent.response = result["text"]
            agent.tokens_used = result["tokens"]
            agent.latency = result["latency"]
            agent.error = False
        except Exception as e:
            agent.response = ""
            agent.error = True
            agent.latency = 0.0
            if self.config.__dict__.get("verbose"):
                console.print(f"[red]Agent {agent.id} ({agent.provider}) error: {e}[/red]")
        return agent

    def _run_iteration(self, prompt: str, iteration: int) -> dict:
        console.print(f"\n[bold cyan]⟳  Iteration {iteration+1}/{self.config.iterations}[/bold cyan]")

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {executor.submit(self._run_agent, agent, prompt): agent
                       for agent in self.agents}

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[yellow]Running {len(self.agents)} agents in parallel...",
                    total=len(self.agents),
                )
                completed = []
                for future in as_completed(futures):
                    completed.append(future.result())
                    progress.advance(task)

        # Score
        all_responses = [a.response for a in completed if a.response]
        for agent in completed:
            agent.score = score_response(agent.response, prompt, all_responses)
            if agent.score > agent.best_score:
                agent.best_score = agent.score
                agent.best_response = agent.response

        completed.sort(key=lambda a: a.score, reverse=True)
        self.agents = completed

        # Update global best
        best = completed[0]
        if best.score > self.global_best_score:
            self.global_best_score = best.score
            self.global_best_response = best.response
            self.global_best_temp = best.temperature

        self._print_table(completed, iteration)

        # PSO update temperatures
        for agent in self.agents:
            agent.temperature = pso_update_temperature(agent, self.global_best_temp, self.config)

        # Pick synthesis client (prefer lowest-latency available)
        synth_client = list(self.clients.values())[0]

        aggregated = aggregate_responses(completed, prompt, synth_client, self.config.max_tokens)

        stats = {
            "iteration": iteration + 1,
            "best_score": best.score,
            "best_temp": best.temperature,
            "best_provider": best.provider,
            "avg_score": round(sum(a.score for a in completed) / len(completed), 4),
            "global_best_score": self.global_best_score,
            "aggregated": aggregated,
            "total_tokens": sum(a.tokens_used for a in completed),
            "errors": sum(1 for a in completed if a.error),
        }
        self.history.append(stats)
        return stats

    def _print_table(self, agents: list[Agent], iteration: int):
        table = Table(
            title=f"Agent Scores Iteration {iteration+1}",
            header_style="bold magenta",
            show_lines=False,
        )
        table.add_column("ID", width=4, style="dim")
        table.add_column("Provider", width=10)
        table.add_column("Temp", width=6)
        table.add_column("Score", width=8)
        table.add_column("Tokens", width=7)
        table.add_column("Latency", width=8)
        table.add_column("Preview", width=55)

        for agent in agents[:12]:
            color = PROVIDER_COLORS.get(agent.provider, "white")
            score_color = "green" if agent.score > 0.6 else "yellow" if agent.score > 0.3 else "red"
            if agent.error:
                preview = "[red]ERROR[/red]"
            else:
                preview = (agent.response[:60] + "…") if len(agent.response) > 60 else agent.response
                preview = preview.replace("\n", " ")

            table.add_row(
                str(agent.id),
                f"[{color}]{agent.provider}[/{color}]",
                str(agent.temperature),
                f"[{score_color}]{agent.score:.3f}[/{score_color}]",
                str(agent.tokens_used),
                f"{agent.latency:.1f}s",
                preview,
            )

        console.print(table)

    def run(self, prompt: str) -> str:
        provider_list = ", ".join(
            f"[{PROVIDER_COLORS.get(p,'white')}]{p}[/{PROVIDER_COLORS.get(p,'white')}]"
            for p in self.clients
        )
        console.print(Panel.fit(
            f"[bold white]🐝  SwarmGPT[/bold white]\n"
            f"Providers: {provider_list}  |  "
            f"Agents: [cyan]{self.config.num_agents}[/cyan]  |  "
            f"Iterations: [cyan]{self.config.iterations}[/cyan]\n"
            f"[dim]{prompt[:90]}{'…' if len(prompt) > 90 else ''}[/dim]",
            border_style="cyan",
        ))

        self._build_agents()

        final_aggregate = ""
        for i in range(self.config.iterations):
            stats = self._run_iteration(prompt, i)
            final_aggregate = stats["aggregated"]

            console.print(
                f"  [green]✓[/green] Iter {i+1} | "
                f"Best: [bold]{stats['best_score']:.3f}[/bold] "
                f"([{PROVIDER_COLORS.get(stats['best_provider'],'white')}]{stats['best_provider']}[/{PROVIDER_COLORS.get(stats['best_provider'],'white')}]) | "
                f"Global best: [bold]{stats['global_best_score']:.3f}[/bold] | "
                f"Tokens: {stats['total_tokens']} | "
                f"Errors: {stats['errors']}"
            )

            if self.global_best_score >= 0.92:
                console.print("[green]✓ Convergence threshold reached stopping early.[/green]")
                break

        console.print("\n")
        console.print(Panel(
            f"{final_aggregate}",
            title="[bold white]🏆  SwarmGPT Final Synthesized Answer[/bold white]",
            border_style="green",
        ))

        self._print_summary()
        return final_aggregate

    def _print_summary(self):
        table = Table(title="Swarm Convergence Summary", header_style="bold blue")
        table.add_column("Iter")
        table.add_column("Best Score")
        table.add_column("Best Provider")
        table.add_column("Avg Score")
        table.add_column("Global Best")
        table.add_column("Tokens")
        table.add_column("Errors")

        for h in self.history:
            color = PROVIDER_COLORS.get(h["best_provider"], "white")
            table.add_row(
                str(h["iteration"]),
                f"{h['best_score']:.3f}",
                f"[{color}]{h['best_provider']}[/{color}]",
                f"{h['avg_score']:.3f}",
                f"{h['global_best_score']:.3f}",
                str(h["total_tokens"]),
                str(h["errors"]),
            )
        console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Provider Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_clients(providers: list[str]) -> dict:
    clients = {}

    if "ollama" in providers:
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "tinyllama")
        client = OllamaClient(url, model)
        if client.check():
            clients["ollama"] = client
            console.print(f"[cyan]✓ Ollama[/cyan] — {model} @ {url}")
        else:
            console.print("[red]✗ Ollama unavailable — skipping[/red]")

    if "openai" in providers:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-..."):
            console.print("[red]✗ OpenAI — OPENAI_API_KEY not set in .env[/red]")
        else:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            max_c = int(os.getenv("OPENAI_MAX_CONCURRENT", "20"))
            client = OpenAIClient(api_key, model, max_c)
            if client.check():
                clients["openai"] = client
                console.print(f"[green]✓ OpenAI[/green] — {model} (max {max_c} concurrent)")
            else:
                console.print("[red]✗ OpenAI check failed — skipping[/red]")

    if "anthropic" in providers:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key or api_key.startswith("sk-ant-..."):
            console.print("[red]✗ Anthropic — ANTHROPIC_API_KEY not set in .env[/red]")
        else:
            model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            max_c = int(os.getenv("ANTHROPIC_MAX_CONCURRENT", "15"))
            client = AnthropicClient(api_key, model, max_c)
            if client.check():
                clients["anthropic"] = client
                console.print(f"[magenta]✓ Anthropic[/magenta] — {model} (max {max_c} concurrent)")
            else:
                console.print("[red]✗ Anthropic check failed — skipping[/red]")

    return clients


def parse_weights(raw: str, active_providers: list[str]) -> dict:
    """Parse 'ollama:10,openai:5' into {'ollama': 10, 'openai': 5}."""
    weights = {}
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                p, w = part.split(":", 1)
                if p.strip() in active_providers:
                    weights[p.strip()] = int(w.strip())
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SwarmGPT Multi-Provider Parallel Swarm Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ollama only (default)
  python swarmgpt.py --prompt "What is entropy?"

  # OpenAI + Ollama mixed swarm
  python swarmgpt.py --prompt "Explain backprop" --providers ollama openai --agents 20

  # Anthropic only, 3 iterations
  python swarmgpt.py --prompt "Write a haiku about AI" --providers anthropic --agents 10 -i 3

  # All three providers, custom weights
  python swarmgpt.py --prompt "..." --providers ollama openai anthropic --weights "ollama:10,openai:5,anthropic:5"
        """,
    )
    parser.add_argument("--prompt", "-p", required=True, help="Target prompt for the swarm")
    parser.add_argument(
        "--providers", nargs="+",
        choices=["ollama", "openai", "anthropic"],
        default=None,
        help="Providers to use (default: from .env or ollama)",
    )
    parser.add_argument("--agents", "-n", type=int, default=None, help="Total number of agents")
    parser.add_argument("--iterations", "-i", type=int, default=None, help="PSO iterations")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens per response")
    parser.add_argument("--temp-min", type=float, default=None)
    parser.add_argument("--temp-max", type=float, default=None)
    parser.add_argument("--weights", type=str, default=None, help='Provider weights e.g. "ollama:10,openai:5"')
    parser.add_argument("--output", "-o", default=None, help="Save final answer to file")
    args = parser.parse_args()

    # Merge CLI args with .env defaults
    providers = args.providers or ["ollama"]
    num_agents = args.agents or int(os.getenv("SWARM_AGENTS", "20"))
    iterations = args.iterations or int(os.getenv("SWARM_ITERATIONS", "5"))
    max_tokens = args.max_tokens or int(os.getenv("SWARM_MAX_TOKENS", "300"))
    temp_min = args.temp_min or float(os.getenv("SWARM_TEMP_MIN", "0.3"))
    temp_max = args.temp_max or float(os.getenv("SWARM_TEMP_MAX", "1.2"))
    weights_raw = args.weights or os.getenv("SWARM_PROVIDER_WEIGHTS", "")

    console.print("\n[bold]SwarmGPT initializing providers...[/bold]")
    clients = build_clients(providers)

    if not clients:
        console.print("[red bold]No providers available. Check your .env and connections.[/red bold]")
        return

    weights = parse_weights(weights_raw, list(clients.keys()))

    config = SwarmConfig(
        providers=list(clients.keys()),
        num_agents=num_agents,
        iterations=iterations,
        temperature_range=(temp_min, temp_max),
        max_tokens=max_tokens,
        provider_weights=weights,
    )

    swarm = SwarmGPT(config, clients)
    result = swarm.run(args.prompt)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        console.print(f"\n[green]✓ Saved to: {args.output}[/green]")


if __name__ == "__main__":
    main()
