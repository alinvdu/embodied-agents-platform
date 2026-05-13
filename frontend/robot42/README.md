# Robot42 Frontend

Vite React UI for Robot42.

## Run

```bash
cd frontend/robot42
npm install
npm run dev
```

Run the agent backend in another terminal:

```bash
python examples/robot42_agent_backend.py \
  --home-memory-path /path/to/house_v1.home_memory.json
```

The backend defaults to `mock`/dry-run mode. To use the OpenAI Agents SDK with a real model:

```bash
python examples/robot42_agent_backend.py \
  --home-memory-path /path/to/house_v1.home_memory.json \
  --provider openai \
  --model gpt-5.5
```

Default backend targets:

- Agent UI server: `http://127.0.0.1:8765`
- Exploration review server: `http://127.0.0.1:8770`

Override them with:

```bash
VITE_AGENT_API_TARGET=http://127.0.0.1:8765 \
VITE_EXPLORATION_API_TARGET=http://127.0.0.1:8770 \
npm run dev
```
