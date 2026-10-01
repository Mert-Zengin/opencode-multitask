import * as Tool from "./tool"
import DESCRIPTION from "./multitask.txt"
import { Session } from "@/session/session"
import { SessionID, MessageID } from "../session/schema"
import { MessageV2 } from "../session/message-v2"
import { Agent } from "../agent/agent"
import { deriveSubagentSessionPermission } from "../agent/subagent-permissions"
import type { TaskPromptOps } from "./task"
import { Config } from "@/config/config"
import { Effect, Schema, Scope } from "effect"
import { Database } from "@opencode-ai/core/database/database"

const id = "multitask"

const SubtaskItem = Schema.Struct({
  id: Schema.optional(Schema.String).annotate({
    description: "An optional short identifier for this task (e.g. 'auth-api', 'frontend-tests')",
  }),
  description: Schema.String.annotate({
    description: "A short (3-5 words) description of this specific sub-task",
  }),
  prompt: Schema.String.annotate({
    description: "The autonomous task instructions for the subagent to perform",
  }),
  subagent_type: Schema.String.annotate({
    description: "The type of specialized agent to use (e.g. 'general', 'explore')",
  }),
})

export const Parameters = Schema.Struct({
  tasks: Schema.Array(SubtaskItem).annotate({
    description: "The list of independent or loosely coupled sub-tasks to run in parallel",
  }),
  concurrency: Schema.optional(Schema.Number).annotate({
    description: "Maximum number of subagents to execute concurrently (default: 4)",
  }),
})

interface SubtaskExecutionResult {
  id: string
  sessionID: SessionID
  description: string
  subagentType: string
  state: "completed" | "error"
  text: string
}

function renderMultitaskOutput(results: SubtaskExecutionResult[]): string {
  const succeeded = results.filter((r) => r.state === "completed").length
  const failed = results.filter((r) => r.state === "error").length

  const items = results.map((r) => {
    const tag = r.state === "error" ? "task_error" : "task_result"
    return [
      `  <task id="${r.id}" session_id="${r.sessionID}" subagent="${r.subagentType}" state="${r.state}">`,
      `    <description>${r.description}</description>`,
      `    <${tag}>`,
      r.text,
      `    </${tag}>`,
      `  </task>`,
    ].join("\n")
  })

  return [
    `<multitask_summary total="${results.length}" succeeded="${succeeded}" failed="${failed}">`,
    ...items,
    `</multitask_summary>`,
  ].join("\n")
}

export const MultitaskTool = Tool.define(
  id,
  Effect.gen(function* () {
    const agent = yield* Agent.Service
    const config = yield* Config.Service
    const sessions = yield* Session.Service
    const database = yield* Database.Service

    const run = Effect.fn("MultitaskTool.execute")(function* (
      params: Schema.Schema.Type<typeof Parameters>,
      ctx: Tool.Context,
    ) {
      if (params.tasks.length === 0) {
        return yield* Effect.fail(new Error("At least one sub-task must be specified in 'tasks'"))
      }

      const cfg = yield* config.get()
      const parent = yield* sessions.get(ctx.sessionID)
      let current = parent
      let depth = 0
      while (current.parentID) {
        depth++
        current = yield* sessions.get(current.parentID)
      }
      if (depth >= (cfg.subagent_depth ?? 1)) {
        return yield* Effect.fail(
          new Error(
            `Subagent depth limit reached (${cfg.subagent_depth ?? 1}). Cannot spawn nested subagents.`,
          ),
        )
      }

      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) {
        return yield* Effect.fail(new Error("MultitaskTool requires promptOps in ctx.extra"))
      }

      const msg = yield* MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID }).pipe(
        Effect.provideService(Database.Service, database),
        Effect.orDie,
      )
      if (msg.info.role !== "assistant") return yield* Effect.fail(new Error("Not an assistant message"))
      const variant = msg.info.variant

      const taskExecutions = params.tasks.map((taskItem, index) =>
        Effect.gen(function* () {
          const taskId = taskItem.id ?? `task-${index + 1}`
          const next = yield* agent.get(taskItem.subagent_type)
          if (!next) {
            return {
              id: taskId,
              sessionID: ctx.sessionID,
              description: taskItem.description,
              subagentType: taskItem.subagent_type,
              state: "error" as const,
              text: `Unknown agent type: ${taskItem.subagent_type}`,
            }
          }

          const childPermission = deriveSubagentSessionPermission({
            parentSessionPermission: parent.permission ?? [],
            subagent: next,
          })

          const childToolDenies = [
            ...(next.permission.some((rule) => rule.permission === "todowrite")
              ? []
              : [{ permission: "todowrite" as const, pattern: "*" as const, action: "deny" as const }]),
            { permission: id as any, pattern: "*" as const, action: "deny" as const },
            { permission: "task" as any, pattern: "*" as const, action: "deny" as const },
          ]

          const nextSession = yield* sessions.create({
            parentID: ctx.sessionID,
            title: `${taskItem.description} (@${next.name} multitask)`,
            agent: next.name,
            permission: [
              ...childPermission,
              ...childToolDenies.filter(
                (deny) =>
                  !childPermission.some(
                    (rule) =>
                      rule.permission === deny.permission && rule.pattern === deny.pattern && rule.action === deny.action,
                  ),
              ),
            ],
          })

          const model = next.model ?? {
            modelID: msg.info.modelID,
            providerID: msg.info.providerID,
          }

          const runSingle = Effect.gen(function* () {
            const parts = yield* ops.resolvePromptParts(taskItem.prompt)
            const result = yield* ops.prompt({
              messageID: MessageID.ascending(),
              sessionID: nextSession.id,
              model: {
                modelID: model.modelID,
                providerID: model.providerID,
              },
              variant: next.model ? undefined : variant,
              agent: next.name,
              parts,
            })

            if (result.info.role === "assistant" && result.info.error) {
              const message =
                "message" in result.info.error.data && typeof result.info.error.data.message === "string"
                  ? result.info.error.data.message
                  : result.info.error.name
              return yield* Effect.fail(new Error(`Subagent failed: ${message}`))
            }

            const failed = result.parts.findLast((item) => item.type === "tool" && item.state.status === "error")
            if (failed?.type === "tool" && failed.state.status === "error") {
              return yield* Effect.fail(new Error(`Subagent tool failed: ${failed.state.error}`))
            }

            const outputText = result.parts.findLast((item) => item.type === "text")?.text ?? ""
            return outputText
          })

          const outcome = yield* runSingle.pipe(
            Effect.map((text) => ({
              id: taskId,
              sessionID: nextSession.id,
              description: taskItem.description,
              subagentType: next.name,
              state: "completed" as const,
              text,
            })),
            Effect.catchAll((err) =>
              Effect.succeed({
                id: taskId,
                sessionID: nextSession.id,
                description: taskItem.description,
                subagentType: next.name,
                state: "error" as const,
                text: err instanceof Error ? err.message : String(err),
              }),
            ),
          )

          return outcome
        }),
      )

      const concurrency = params.concurrency && params.concurrency > 0 ? params.concurrency : 4
      const results = yield* Effect.all(taskExecutions, { concurrency })

      const summaryText = renderMultitaskOutput(results)
      const overallTitle = `Multitask: ${params.tasks.length} subagents (${results.filter((r) => r.state === "completed").length}/${results.length} succeeded)`

      yield* ctx.metadata({
        title: overallTitle,
        metadata: {
          total: results.length,
          succeeded: results.filter((r) => r.state === "completed").length,
          failed: results.filter((r) => r.state === "error").length,
          tasks: results.map((r) => ({
            id: r.id,
            sessionId: r.sessionID,
            status: r.state,
          })),
        },
      })

      return {
        title: overallTitle,
        metadata: {
          taskCount: results.length,
          tasks: results.map((r) => ({ id: r.id, sessionId: r.sessionID, status: r.state })),
        },
        output: summaryText,
      }
    })

    return {
      description: DESCRIPTION,
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        run(params, ctx).pipe(Effect.orDie),
    }
  }),
)
