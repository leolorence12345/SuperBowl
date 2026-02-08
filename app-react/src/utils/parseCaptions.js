/**
 * Parse caption text into { start_sec, end_sec, text }[].
 * Supports: JSON array, or lines like "10:00 - 10:20 - Event" / "10:05 - Event" / "600 - 620 - Event"
 */
function parseTimeToSeconds(s) {
  if (!s || typeof s !== 'string') return 0
  s = s.trim()
  if (!s) return 0
  if (/^\d+$/.test(s)) return parseInt(s, 10)
  const parts = s.split(':')
  if (parts.length === 2) {
    const m = parseInt(parts[0].trim(), 10)
    const sec = parseInt(parts[1].trim(), 10)
    return m * 60 + sec
  }
  if (parts.length === 3) {
    const h = parseInt(parts[0].trim(), 10)
    const m = parseInt(parts[1].trim(), 10)
    const sec = parseInt(parts[2].trim(), 10)
    return h * 3600 + m * 60 + sec
  }
  return 0
}

const TIME_PART = /^(\d{1,3}:\d{2}(?::\d{2})?|\d+)\s*-\s*(\d{1,3}:\d{2}(?::\d{2})?|\d+)\s*-\s*(.+)$/
const TIME_SINGLE = /^(\d{1,3}:\d{2}(?::\d{2})?|\d+)\s*-\s*(.+)$/

export function parseCaptions(text) {
  const out = []
  if (!text || typeof text !== 'string') return out
  const raw = text.trim()
  if (!raw) return out

  // Try JSON
  try {
    const data = JSON.parse(raw)
    if (Array.isArray(data)) {
      for (const item of data) {
        if (item && typeof item === 'object') {
          let start = item.start ?? item.start_sec ?? 0
          let end = item.end ?? item.end_sec ?? (Number(start) + 20)
          const t = item.text ?? item.content ?? ''
          if (typeof start === 'string') start = parseTimeToSeconds(start)
          if (typeof end === 'string') end = parseTimeToSeconds(end)
          out.push({ start_sec: start | 0, end_sec: end | 0, text: String(t) })
        }
      }
      return out
    }
  } catch (_) {}

  // Line-based
  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const m1 = trimmed.match(TIME_PART)
    if (m1) {
      out.push({
        start_sec: parseTimeToSeconds(m1[1]),
        end_sec: parseTimeToSeconds(m1[2]),
        text: m1[3].trim(),
      })
      continue
    }
    const m2 = trimmed.match(TIME_SINGLE)
    if (m2) {
      const start = parseTimeToSeconds(m2[1])
      out.push({ start_sec: start, end_sec: start + 20, text: m2[2].trim() })
    }
  }
  return out
}

export function getCaptionAtTime(captions, currentTimeSec) {
  if (!Array.isArray(captions) || !captions.length) return null
  for (const c of captions) {
    if (currentTimeSec >= c.start_sec && currentTimeSec <= c.end_sec) return c.text
  }
  return null
}

export function formatTime(sec) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s < 10 ? '0' : ''}${s}`
}
