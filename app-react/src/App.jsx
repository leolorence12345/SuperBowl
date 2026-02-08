import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const SEGMENT_INTERVAL = 5 // seconds per segment

function App() {
  const [objectUrl, setObjectUrl] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const videoRef = useRef(null)

  // Business config
  const [businessName, setBusinessName] = useState('MVP Pizza')
  const [businessType, setBusinessType] = useState('pizza restaurant')

  // Upload state
  const [videoUri, setVideoUri] = useState(null) // Gemini URI
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')

  // Live events + ads (appended in real time as video plays)
  const [events, setEvents] = useState([])
  const [ads, setAds] = useState([])

  // Track which segments we've already sent for analysis
  const analyzedSegmentsRef = useRef(new Set())
  const [analyzingSegment, setAnalyzingSegment] = useState(null) // "0:00 – 0:05"

  const [copiedIdx, setCopiedIdx] = useState(-1)
  const eventsEndRef = useRef(null)
  const adsEndRef = useRef(null)

  const handleTimeUpdate = useCallback(() => {
    const v = videoRef.current
    if (v && !isNaN(v.currentTime)) setCurrentTime(v.currentTime)
  }, [])

  const formatSec = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  // ── Upload video: save locally in browser + send to backend → Gemini ──
  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Show video locally immediately
    if (objectUrl) URL.revokeObjectURL(objectUrl)
    setObjectUrl(URL.createObjectURL(file))

    // Reset state for fresh run
    setEvents([])
    setAds([])
    analyzedSegmentsRef.current = new Set()
    setVideoUri(null)
    setUploadError('')
    setUploading(true)

    // Send file to backend → Gemini upload
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/api/upload-video`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      setVideoUri(data.video_uri)
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(false)
    }
  }

  // ── Live analysis: as video crosses each 5-sec boundary, call /api/live-segment ──
  useEffect(() => {
    if (!videoUri) return // not ready yet

    // Which segment does currentTime fall in?
    const segStart = Math.floor(currentTime / SEGMENT_INTERVAL) * SEGMENT_INTERVAL
    const segEnd = segStart + SEGMENT_INTERVAL

    // Already analyzed or currently analyzing this segment?
    const segKey = `${segStart}-${segEnd}`
    if (analyzedSegmentsRef.current.has(segKey)) return
    // Also analyze all prior un-analyzed segments (in case user seeked forward)
    const segmentsToAnalyze = []
    for (let s = 0; s <= segStart; s += SEGMENT_INTERVAL) {
      const key = `${s}-${s + SEGMENT_INTERVAL}`
      if (!analyzedSegmentsRef.current.has(key)) {
        segmentsToAnalyze.push({ start: s, end: s + SEGMENT_INTERVAL, key })
      }
    }

    if (segmentsToAnalyze.length === 0) return

    // Mark all as in-progress
    segmentsToAnalyze.forEach(seg => analyzedSegmentsRef.current.add(seg.key))

    // Fire analyses (sequentially to avoid overwhelming the API)
    ;(async () => {
      for (const seg of segmentsToAnalyze) {
        const window = `${Math.floor(seg.start / 60)}:${(seg.start % 60).toString().padStart(2, '0')} – ${Math.floor(seg.end / 60)}:${(seg.end % 60).toString().padStart(2, '0')}`
        setAnalyzingSegment(window)

        try {
          const res = await fetch(`${API_BASE}/api/live-segment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              start_sec: seg.start,
              end_sec: seg.end,
              video_uri: videoUri,
              business_name: businessName,
              business_type: businessType,
            }),
          })
          const data = await res.json()

          // Append event
          if (data.event) {
            setEvents(prev => [...prev, data.event])
          }
          // Append ad
          if (data.ad) {
            setAds(prev => [...prev, { ...data.ad, source_event: data.event }])
          }
        } catch (err) {
          console.error(`Segment ${seg.key} failed:`, err)
        }
      }
      setAnalyzingSegment(null)
    })()
  }, [currentTime, videoUri, businessName, businessType])

  // Auto-scroll
  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  useEffect(() => {
    adsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [ads.length])

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(-1), 2000)
  }

  const urgencyColor = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#22c55e',
  }

  const significantAds = ads.filter(a => a.is_significant)
  const isAnalyzing = analyzingSegment !== null

  return (
    <div className="dashboard">
      {/* Top bar */}
      <header className="top-bar">
        <div className="top-bar-brand">
          <h1>SuperBowl Ad Pulse</h1>
          <span className="tagline">Real-time AI ads from live game moments</span>
        </div>
        <div className="top-bar-config">
          <input
            type="text"
            placeholder="Business name"
            value={businessName}
            onChange={(e) => setBusinessName(e.target.value)}
            className="config-input"
          />
          <input
            type="text"
            placeholder="Business type"
            value={businessType}
            onChange={(e) => setBusinessType(e.target.value)}
            className="config-input"
          />
          <div className="live-indicator">
            {(isAnalyzing || uploading) && <span className="live-dot" />}
            <span className="live-text">
              {uploading
                ? 'Uploading to Gemini...'
                : isAnalyzing
                  ? `Analyzing ${analyzingSegment}...`
                  : `${significantAds.length} ad${significantAds.length !== 1 ? 's' : ''} ready`}
            </span>
          </div>
        </div>
      </header>

      <div className="main-grid">
        {/* Left: Video + Events */}
        <div className="left-col">
          <div className="video-card">
            <div className="video-container">
              {objectUrl ? (
                <video
                  ref={videoRef}
                  src={objectUrl}
                  controls
                  onTimeUpdate={handleTimeUpdate}
                  onSeeked={handleTimeUpdate}
                  onPlay={handleTimeUpdate}
                />
              ) : (
                <div className="video-placeholder">
                  <label className="upload-label">
                    {uploading ? 'Uploading...' : 'Upload game video'}
                    <input type="file" accept="video/*" onChange={handleFileChange} hidden disabled={uploading} />
                  </label>
                </div>
              )}
            </div>
            {/* Status bar under video */}
            {objectUrl && (
              <div className="video-info-bar">
                <span className="time-display">{formatSec(currentTime)}</span>
                {uploading && <span className="upload-status">Uploading to Gemini...</span>}
                {!uploading && videoUri && (
                  <span className="upload-status ready">Gemini ready</span>
                )}
                {!uploading && !videoUri && !uploadError && (
                  <span className="upload-status">Waiting for upload...</span>
                )}
                {uploadError && <span className="upload-status error">{uploadError}</span>}
                <span className="event-counter">{events.length} events | {significantAds.length} ads</span>
              </div>
            )}
          </div>

          {/* Events timeline — populated LIVE as Gemini analyzes each segment */}
          <div className="events-card">
            <h3>Live Game Events ({events.length})</h3>
            <div className="events-list">
              {events.map((ev, i) => {
                const adForEvent = ads.find(a =>
                  a.source_event?.start_sec === ev.start_sec && a.source_event?.end_sec === ev.end_sec
                )
                const hasAd = adForEvent?.is_significant
                return (
                  <div
                    key={`${ev.start_sec}-${ev.end_sec}`}
                    className={`event-item ${
                      currentTime >= ev.start_sec && currentTime < ev.end_sec ? 'active' : ''
                    } ${i === events.length - 1 ? 'newest' : ''}`}
                    onClick={() => {
                      if (videoRef.current) videoRef.current.currentTime = ev.start_sec
                    }}
                  >
                    <span className="event-time">{ev.window}</span>
                    <span className="event-desc">
                      {ev.analysis?.slice(0, 120) || '(no events)'}
                    </span>
                    {hasAd && <span className="event-ad-dot" title="Ad generated" />}
                  </div>
                )
              })}
              {/* Loading indicator for current analysis */}
              {isAnalyzing && (
                <div className="event-item analyzing">
                  <span className="event-time">{analyzingSegment}</span>
                  <span className="event-desc analyzing-text">Analyzing with Gemini...</span>
                </div>
              )}
              <div ref={eventsEndRef} />
              {events.length === 0 && !isAnalyzing && (
                <p className="empty-msg">
                  {videoUri
                    ? 'Play the video — Gemini will analyze each segment live.'
                    : uploading
                      ? 'Uploading video to Gemini...'
                      : 'Upload a video to get started.'}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Right: Key Moments table + Ad Feed */}
        <div className="right-col">
          {/* ── Key Moments highlights table ── */}
          {significantAds.length > 0 && (
            <div className="key-moments-card">
              <h3>Key Moments</h3>
              <table className="key-moments-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Tag</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {significantAds.map((ad, i) => (
                    <tr
                      key={i}
                      className="key-moment-row"
                      onClick={() => {
                        if (videoRef.current && ad.source_event?.start_sec != null) {
                          videoRef.current.currentTime = ad.source_event.start_sec
                        }
                      }}
                    >
                      <td className="km-time">{ad.source_event?.window || '—'}</td>
                      <td>
                        <span className={`km-tag km-tag-${ad.event_type || 'other'}`}>
                          {ad.event_type || 'play'}
                        </span>
                      </td>
                      <td className="km-desc">{ad.ad_copy?.slice(0, 60) || ad.source_event?.analysis?.slice(0, 60) || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* ── Ad Feed ── */}
          <div className="ads-card">
            <h3>
              Live Ad Feed
              {significantAds.length > 0 && (
                <span className="ad-count">{significantAds.length} ads</span>
              )}
            </h3>

            <div className="ads-list">
              {ads.map((ad, i) => {
                if (!ad.is_significant) return null
                return (
                  <div key={i} className="ad-card ad-enter">
                    <div className="ad-header">
                      <span
                        className="urgency-badge"
                        style={{ background: urgencyColor[ad.urgency] || '#666' }}
                      >
                        {ad.urgency || 'low'}
                      </span>
                      <span className="event-type-badge">{ad.event_type || 'play'}</span>
                      <span className="ad-time">
                        {ad.source_event?.window || ''}
                      </span>
                    </div>
                    <div className="ad-copy">{ad.ad_copy}</div>
                    <div className="ad-promo">{ad.promo_suggestion}</div>
                    {ad.social_hashtags?.length > 0 && (
                      <div className="ad-hashtags">
                        {ad.social_hashtags.map((h, j) => (
                          <span key={j} className="hashtag">{h}</span>
                        ))}
                      </div>
                    )}
                    <div className="ad-actions">
                      <button
                        className="copy-btn"
                        onClick={() =>
                          copyToClipboard(
                            `${ad.ad_copy}\n\n${ad.promo_suggestion}\n\n${(ad.social_hashtags || []).join(' ')}`,
                            i
                          )
                        }
                      >
                        {copiedIdx === i ? 'Copied!' : 'Copy ad'}
                      </button>
                    </div>
                  </div>
                )
              })}
              <div ref={adsEndRef} />
              {ads.length === 0 && (
                <p className="empty-msg">
                  {videoUri
                    ? 'Play the video — ads appear as game moments are detected.'
                    : 'Upload a video to get started.'}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
