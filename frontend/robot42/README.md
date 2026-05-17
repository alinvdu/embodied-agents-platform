# Robot42 Frontend

Vite React UI for Robot42.

Full workflow: [Robot42 Agentic End To End](../../command_reference/robot42_agentic_end_to_end.md).

## Run

```bash
cd frontend/robot42
npm install
npm run dev
```

Run the agent backend in another terminal:

```bash
python examples/robot42_agent_backend.py \
  --memory-root /Users/alindumitru/Robot42/artifacts/memories
```

You can still pass `--home-memory-path` for one exact memory, but the normal Robot42 flow is to let the backend discover environment folders under `artifacts/memories`.

The backend defaults to `mock`/dry-run mode. To use the OpenAI Agents SDK with a real model:

```bash
python examples/robot42_agent_backend.py \
  --memory-root /Users/alindumitru/Robot42/artifacts/memories \
  --provider openai \
  --model gpt-5.5
```

Run the real exploration backend when you need to configure or update the environment:

```bash
python -m xlerobot_playground.real_agentic_exploration \
  --memory-root /Users/alindumitru/Robot42/artifacts/memories
```

`examples/xlerobot_exploration_review.py` is only a review-only helper for already-saved maps. The normal robot mapping flow is `python -m xlerobot_playground.real_agentic_exploration`.

If memory already exists and you only want the agent to reason over memory or produce navigation previews, do not start exploration. Start only `robot42_agent_backend.py` and the React UI. Start `real_agentic_exploration` when you need `Configure Environment` to create, load, edit, or re-approve an environment map, or when you want the agent to execute live Nav2 waypoint navigation/relocalization.

For live navigation, run the agent backend with a live model provider:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --provider openai \
  --model gpt-5.5 \
  --exploration-backend-url http://127.0.0.1:8770 \
  --navigation-waypoint-horizon-m 2.0
```

Without `--provider openai`, the backend defaults to `mock/mock`; it only resolves previews and will not move the robot or create OpenAI Agent traces.

To load a saved environment in the configuration UI:

1. Start `real_agentic_exploration` with the same `--memory-root`.
2. Open `Configure Environment`.
3. Select the saved environment.
4. Click `Load`.
5. Click `Start Nav Session` when you want live ROS/Nav2 manual waypoint testing on that saved map.

`Start Nav Session` uses the loaded `environment_map.json`, applies the saved dock/start pose as the initial robot pose, and enables the existing `Preview` / `Go` controls without starting frontier exploration.

Agent navigation uses the same exploration backend calls as the UI:

- `navigate_to_waypoint` posts to `/api/nav/waypoint`
- `relocalize_here` posts to `/api/nav/relocalize`

Keep `real_agentic_exploration` running with a loaded environment and an active nav session when you want those agent tools to move the robot.

Default backend targets:

- Agent UI server: `http://127.0.0.1:8765`
- Exploration review server: `http://127.0.0.1:8770`

Override them with:

```bash
VITE_AGENT_API_TARGET=http://127.0.0.1:8765 \
VITE_EXPLORATION_API_TARGET=http://127.0.0.1:8770 \
npm run dev
```
