# adk-python — Upstream Contribution Fork

This is my working fork of [google/adk-python](https://github.com/google/adk-python),
Google's Agent Development Kit. I maintain it to land fixes and features upstream;
this README tracks that work.

For the actual project, README, and installation instructions, see
[google/adk-python](https://github.com/google/adk-python).

## Upstream Contributions

### In review
- [#5683](https://github.com/google/adk-python/pull/5683) — `fix: terminate infinite retry loop in RunSkillScriptTool on SCRIPT_NOT_FOUND`
- [#5651](https://github.com/google/adk-python/pull/5651) — `fix: terminate infinite retry loop in LoadSkillResourceTool on RESOURCE_NOT_FOUND`
- [#5023](https://github.com/google/adk-python/pull/5023) — `fix: raise ValueError for unsupported MIME types in file_data URI path`
- [#4748](https://github.com/google/adk-python/pull/4748) — `fix(sessions): prevent PydanticSerializationError when session state contains non-serializable objects`

### Closed
- [#4806](https://github.com/google/adk-python/pull/4806) — `fix(tools): support Python 3.10+ pipe union syntax in function parameter parser`
- [#4736](https://github.com/google/adk-python/pull/4736) — `feat(memory): add DatabaseMemoryService with SQL backend and agent scratchpad` (+1855 LOC)
- [#3002](https://github.com/google/adk-python/pull/3002) — `fix: improve OpenAPI operation parser type safety and code organization`
- [#2768](https://github.com/google/adk-python/pull/2768) — `Add SQLite-based memory service implementation`

## Focus Areas
Tool error handling and retry semantics · session serialization · memory service backends · OpenAPI integration · type-safety fixes.
