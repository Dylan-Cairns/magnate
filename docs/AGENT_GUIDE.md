# Memory Bank Operations Guide

Start in `AGENTS.md` for quick-start rituals. This guide covers the workflows for keeping the Memory Bank accurate for GPT-5 Codex sessions.

The Codex CLI agent is stateless between sessions. To maintain continuity, keep precise documentation in this repo’s Memory Bank. At the start of every substantive task, the agent should review the Memory Bank to rehydrate project context.

## Memory Bank Structure

The Memory Bank consists of core files and optional context files, all in Markdown format. Files build upon each other in a clear hierarchy:

flowchart TD
PB[projectBrief.md] --> SP[systemPatterns.md]
PB --> TC[techContext.md]

MR[magnateRules.md] --> SP
MR --> AC[activeContext.md]

SP --> AC[activeContext.md]
TC --> AC

AC --> P[progress.md]

### Core Files (Required)

1. `projectBrief.md`

   - Foundation document that shapes all other files
   - Defines core requirements and goals
   - Source of truth for project scope

2. `activeContext.md`

   - Current work focus
   - Recent changes
   - Next steps
   - Active decisions and considerations
   - Important patterns and preferences
   - Learnings and project insights

3. `systemPatterns.md`

   - System architecture
   - Key technical decisions
   - Design patterns in use
   - Component relationships
   - Critical implementation paths

4. `techContext.md`

   - Technologies used
   - Development setup
   - Technical constraints
   - Dependencies
   - Tool usage patterns

5. `progress.md`

   - What works
   - What's left to build
   - Current status
   - Known issues
   - Evolution of project decisions

6. `magnateRules.md`
   - Official rules reference
   - Source of truth for engine legality, income, placement, scoring, and all other game rules

### Additional Context

Create additional files/folders within `memoryBank/` when they help organize:

- Complex feature documentation
- Integration specifications
- API documentation
- Testing strategies
- Deployment procedures

Current additional context file:
- `bridgeInterfaceContract.md` (TS<->Python boundary contract)

## Core Workflows

### Plan Mode

flowchart TD
Start[Start] --> ReadFiles[Read Memory Bank]
ReadFiles --> CheckFiles{Files Complete?}

    CheckFiles -->|No| Plan[Create Plan]
    Plan --> Document[Document in Chat]

    CheckFiles -->|Yes| Verify[Verify Context]
    Verify --> Strategy[Develop Strategy]
    Strategy --> Present[Present Approach]

Always include `magnateRules.md` when planning or modifying rules, engine state, legality, income, or scoring.

### Act Mode

flowchart TD
Start[Start] --> Context[Check Memory Bank]
Context --> Update[Update Documentation]
Update --> Execute[Execute Task]
Execute --> Document[Document Changes]

Verify any rules work against `magnateRules.md` before implementation.

## Documentation Updates

Memory Bank updates occur when:

1. Discovering new project patterns
2. After implementing significant changes
3. When the user requests with "update memory bank" (review ALL files)
4. When context needs clarification
5. When architecture decisions change (update `README.md`, `AGENTS.md`, and relevant Memory Bank files in the same pass)

flowchart TD
Start[Update Process]

    subgraph Process
        P1[Review ALL Files]
        P2[Document Current State]
        P3[Clarify Next Steps]
        P4[Document Insights & Patterns]

        P1 --> P2 --> P3 --> P4
    end

    Start --> Process

Note: When triggered by "update memory bank", the agent should review every Memory Bank file, even if some don't require edits. Focus particularly on `activeContext.md` and `progress.md` as they track current state.

REMEMBER: Because sessions are stateless, the Memory Bank is the primary link to previous work. Maintain it with precision and clarity.
