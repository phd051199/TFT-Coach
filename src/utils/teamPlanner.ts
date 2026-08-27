import type { Champion } from '../types'

const TEAM_PLANNER_FORMAT_VERSION = '02'
const TEAM_PLANNER_SLOTS = 10

export function encodeTeamPlannerCode(
  champions: Champion[],
  allChampions: Champion[],
  setNumber = 18,
) {
  const plannerCodeById = new Map(
    allChampions.map((champion, index) => [
      champion.id,
      champion.teamPlannerCode ?? 1000 + index,
    ]),
  )

  const slots = champions.slice(0, TEAM_PLANNER_SLOTS).map((champion) => {
    const code = plannerCodeById.get(champion.id)
    if (code === undefined || code < 0 || code > 0xfff) {
      throw new Error(`Invalid Team Planner code for ${champion.id}`)
    }
    return code.toString(16).padStart(3, '0')
  })

  while (slots.length < TEAM_PLANNER_SLOTS) slots.push('000')
  return `${TEAM_PLANNER_FORMAT_VERSION}${slots.join('')}TFTSet${setNumber}`
}
