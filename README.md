# Embodied Agents Platform

ROS 2 platform for embodied AI agents that perceive, plan, and act in the physical world.

This repository brings together a real XLeRobot, RGB-D perception, ROS 2/Nav2 navigation, map building, semantic memory, and LLM-driven decision making. The goal is to turn a mobile robot from a teleoperated device into an agent that can explore a home, remember what it saw, reason over that memory, and execute physical navigation and inspection tasks.

## What It Does

- Runs a real robot through ROS 2, Nav2, `/cmd_vel`, odometry, RGB-D, point clouds, and occupancy maps.
- Builds maps from camera/head scans, fused scan data, point clouds, and OctoMap projected maps.
- Detects frontiers, previews candidate paths through Nav2, and chooses where to explore next.
- Supports both heuristic and LLM frontier selection.
- Provides a live web UI for exploration review, map editing, environment approval, and manual waypoint navigation.
- Saves approved environments as reusable memory for later agent tasks.
- Lets an agent reason over saved home memory and call tools such as waypoint navigation, relocalization, region scans, object detection, object centering, and approach behaviors.
- Includes simulation and playground modes so exploration logic can be tested without the physical robot.
- Includes VR and keyboard teleoperation paths for data collection and direct control.

## System Shape

```text
User / Agent UI
      |
      v
Agent backend and tool registry
      |
      v
Exploration backend + memory store
      |
      v
ROS 2 / Nav2 / SLAM / OctoMap
      |
      v
real_ros_bridge -> robot brain -> XLeRobot motors, camera, RGB-D, IMU
```

The real exploration loop is:

```text
scan -> update map -> detect frontiers -> preview Nav2 paths
     -> choose next goal -> navigate_to_pose -> repeat -> save environment memory
```

For real hardware runs, Nav2 performs path planning and local velocity control. The robot brain executes bounded velocity and camera commands on the XLeRobot.

## Capabilities

### Real-World Exploration

The real robot can perform an initial 360 degree camera-pan scan, build an occupancy representation, select frontiers, navigate to them with Nav2, and keep expanding the map. Use [end-to-end-exploration.md](end-to-end-exploration.md) for the concise exploration-only commands; [REAL_EXPLORATION_END_TO_END.md](REAL_EXPLORATION_END_TO_END.md) contains the detailed setup, alternatives, calibration, and troubleshooting notes.

### Navigation and Mapping

The platform includes ROS/Nav2 runtime code, generated Nav2 parameters, SLAM and OctoMap launch files, path previewing, relocalization hooks, RGB-D visual odometry, wheel odometry, scan fusion, point-cloud fusion, and diagnostics for odometry and rotation.

Important modules:

- [xlerobot_playground/real_agentic_exploration.py](xlerobot_playground/real_agentic_exploration.py)
- [xlerobot_playground/ros_nav2_runtime.py](xlerobot_playground/ros_nav2_runtime.py)
- [xlerobot_playground/real_ros_bridge.py](xlerobot_playground/real_ros_bridge.py)
- [xlerobot_playground/nav2_goal_client.py](xlerobot_playground/nav2_goal_client.py)
- [launch/xlerobot_octomap.launch.py](launch/xlerobot_octomap.launch.py)

### Agent Runtime

The agent layer turns saved environment memory into actionable robot tools. It can inspect known areas, preview or execute navigation, scan regions for an object, save what the robot saw, estimate object geometry from RGB-D, and issue small approach movements.

Important modules:

- [xlerobot_agent/home_agent.py](xlerobot_agent/home_agent.py)
- [xlerobot_agent/tools.py](xlerobot_agent/tools.py)
- [xlerobot_agent/home_memory.py](xlerobot_agent/home_memory.py)
- [xlerobot_agent/object_detection.py](xlerobot_agent/object_detection.py)
- [examples/robot42_agent_backend.py](examples/robot42_agent_backend.py)

### Web UI

The React UI can connect to the agent backend and exploration backend. It supports saved environment loading, live navigation sessions, map interaction, and the "What Robot Saw" panel for object-search runs.

See [frontend/robot42/README.md](frontend/robot42/README.md).

### Simulation and Playgrounds

Simulation paths let you develop exploration logic with ManiSkill, synthetic backends, teleport movement, or ROS/Nav2 without needing every hardware component running.

Start with [RUNNING_EXPLORATION.md](RUNNING_EXPLORATION.md) and [SIMULATION_NAV2.md](SIMULATION_NAV2.md).

### Teleoperation and Data Collection

The repo includes keyboard and VR control paths for XLeRobot, plus recording-oriented flows for collecting demonstrations.

See [VR_TELEOPERATION.md](VR_TELEOPERATION.md) and [TELEOPERATION_TRAINING.md](TELEOPERATION_TRAINING.md).

## Quick Start

This project is currently a research/development stack, not a one-command package install. The exact command path depends on whether you are running simulation, ROS/Nav2, or the physical robot.

### Run the Agent UI

```bash
cd frontend/robot42
npm install
npm run dev
```

In another terminal, start the agent backend:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --agent-artifacts-root ./artifacts/agent_runs
```

The backend defaults to mock mode. For a live model-backed agent, pass a provider and model, for example:

```bash
python examples/robot42_agent_backend.py \
  --memory-root ./artifacts/memories \
  --provider openai \
  --model gpt-5.5 \
  --exploration-backend-url http://127.0.0.1:8770 \
  --agent-artifacts-root ./artifacts/agent_runs
```

### Run Simulated Exploration

```bash
python examples/xlerobot_exploration_playground.py \
  --movement-mode simulated \
  --ui-flavor user \
  --explorer-policy heuristic
```

For LLM frontier decisions, use `--explorer-policy llm` and configure the provider/model flags documented in [RUNNING_EXPLORATION.md](RUNNING_EXPLORATION.md).

### Run Real Exploration

The real path expects the robot brain, Orbbec sidecar, real ROS bridge, odometry, Nav2, and mapping stack to be running. The full multi-terminal sequence is in [REAL_EXPLORATION_END_TO_END.md](REAL_EXPLORATION_END_TO_END.md).

The exploration runtime entrypoint is:

```bash
python -m xlerobot_playground.real_agentic_exploration \
  --memory-root ./artifacts/memories
```

## Repository Layout

```text
xlerobot_agent/        Agent runtime, memory, tools, prompts, object detection
xlerobot_playground/   ROS/Nav2 runtimes, bridges, mapping, exploration, diagnostics
multido_xlerobot/      Integration facade for the local XLeRobot/LeRobot fork
examples/              Runnable backend, playground, bridge, and service entrypoints
frontend/robot42/      React UI for agent and exploration workflows
launch/                ROS 2 launch files for OctoMap/relocalization
config/                Mapping and OctoMap configuration
scripts/               Setup, rendering, capture, and diagnostic helpers
tests/                 Unit and integration tests for exploration, ROS, agent tools
plans/                 Design notes and implementation plans
```

## Useful Docs

- [REAL_EXPLORATION_END_TO_END.md](REAL_EXPLORATION_END_TO_END.md): real robot exploration runbook
- [RUNNING_EXPLORATION.md](RUNNING_EXPLORATION.md): simulation, ROS, and playground exploration commands
- [SMOKE_TEST_END_TO_END_EXPLORATION.md](SMOKE_TEST_END_TO_END_EXPLORATION.md): smoke-test workflow
- [SIMULATION_NAV2.md](SIMULATION_NAV2.md): ManiSkill + ROS/Nav2 notes
- [AGENT_RUNTIME.md](AGENT_RUNTIME.md): simplified agent runtime overview
- [INTEGRATION.md](INTEGRATION.md): XLeRobot fork integration boundary
- [ODOMETRY_TESTS.md](ODOMETRY_TESTS.md): odometry validation notes
- [command_reference/README.md](command_reference/README.md): command reference index

## Testing

Run the Python test suite with:

```bash
python -m unittest
```

Focused tests are useful while developing a specific subsystem:

```bash
python -m unittest tests.test_real_agentic_exploration
python -m unittest tests.test_exploration_backend
python -m unittest tests.test_home_agent
```

## Status

This is an active embodied-agent research stack. Several paths are real hardware-backed today, including ROS/Nav2 navigation, RGB-D ingestion, exploration review, saved environment memory, and agent tool execution. Some parts remain experimental or provider-dependent, including live VLM/LLM policies, object detection backends, grasp/VLA execution, and full production-grade safety validation.

Use conservative robot speed limits, keep a physical stop option available, and test new navigation behaviors in simulation or smoke-test mode before running them on hardware.
