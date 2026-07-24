# RC-Bench JMLC Harness

Пакет предназначен для размещения в корне актуальной ветки `dev` репозитория `rc-bench`.

## Состав

- `AGENTS.md` — обязательные правила для Codex и сабагентов;
- `CODEX_START_PROMPT.md` — первый запрос для новой сессии Codex;
- `docs/agent/AUDIT_BASELINE.md` — исходное состояние и подтверждённые дефекты;
- `docs/agent/PROJECT_CONTRACT.md` — неизменяемые параметры результата;
- `docs/agent/WORK_PLAN.md` — порядок работ до защиты;
- `docs/agent/BACKLOG.md` — исполнимые задачи и Definition of Done;
- `docs/agent/DECISIONS.md` — журнал решений;
- `docs/agent/AGENT_WORKFLOW.md` — роли, worktree и интеграция;
- `docs/agent/HANDOFF_TEMPLATE.md` — формат передачи результата;
- `scripts/verify.sh` — quick, smoke и release gates.

## Использование

1. Открой `CODEX_START_PROMPT.md` и передай его Codex в локальном checkout.
2. Сначала проверь и синхронизируй ветки. Не копируй старый `dev` поверх `main`.
3. Размести содержимое пакета в корне актуального `dev`, сохранив структуру каталогов.
4. Установи окружение проекта командой `poetry install`.
5. Сделай `scripts/verify.sh` исполняемым в Linux/macOS: `chmod +x scripts/verify.sh`.
6. Начни с `BOOT-001` и двигайся по зависимостям из backlog.

Сам пакет не содержит изменений вычислительного кода и не должен автоматически запускать полный эксперимент.
