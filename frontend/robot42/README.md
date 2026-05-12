# Robot42 Frontend

Vite React UI for Robot42.

## Run

```bash
cd frontend/robot42
npm install
npm run dev
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
