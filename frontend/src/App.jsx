import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { askQuestionStream, getHealth, warmVlm, warmGuard, sendFeedback } from './api'
import './App.css'

const MAX_BYTES = 10 * 1024 * 1024 // keep in sync with backend MAX_CONTENT_LENGTH

// Per-stage progress rows, in the order they should appear. Keyed by the backend's
// SSE `stage` name (app.py's _ask_events) — vlm_start is intentionally omitted (a
// near-instant no-op for VLM_PROVIDER=cloudrun; showing it would just flicker).
const STAGE_LABELS = [
  { stage: 'chart_gate', label: 'Verifying image' },
  { stage: 'guard', label: 'Verifying question' },
  { stage: 'vlm', label: 'Processing model' },
]

// Baked in at BUILD time (Vite's import.meta.env is compile-time, not runtime — see
// frontend/cloudbuild.yaml / Dockerfile). Empty in local dev by default: no login wall,
// matches the backend's AUTH_ENABLED=0 default (see docs/REVIEW_AND_ROADMAP.md §3.7).
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''
const HAS_AUTH = Boolean(GOOGLE_CLIENT_ID)
const TOKEN_STORAGE_KEY = 'chartqa_gid_token'

// Mirror of the backend's _question_too_weak guard for instant feedback.
// Count letters/digits in any language (so CJK questions pass), reject junk.
function questionTooWeak(q) {
  const meaningful = (q.match(/[\p{L}\p{N}]/gu) || []).length
  return meaningful < 3
}

function App() {
  // The chart the conversation is about — pinned for the whole session (Phase 5).
  const [image, setImage] = useState(null) // File
  const [previewUrl, setPreviewUrl] = useState('')
  const [question, setQuestion] = useState('')
  // The chat transcript. Each message: { role: 'user'|'assistant', text, meta? }.
  // Assistant messages also carry { detect?, mock?, latency_ms?, feedback? } for rendering.
  const [messages, setMessages] = useState([])
  // Server-side conversation id, returned on turn 1 and sent back on every follow-up.
  const [conversationId, setConversationId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  // Per-stage progress for the pending turn, keyed by SSE stage name.
  const [stages, setStages] = useState({})
  const [mockBanner, setMockBanner] = useState(false)
  // Google ID token (Phase 3.7 required sign-in). null when signed out; also null
  // permanently when HAS_AUTH is false. sessionStorage: survives refresh, cleared on tab close.
  const [token, setToken] = useState(() =>
    HAS_AUTH ? sessionStorage.getItem(TOKEN_STORAGE_KEY) : null
  )
  const fileInputRef = useRef(null)
  const abortRef = useRef(null)
  const signinButtonRef = useRef(null)
  const transcriptEndRef = useRef(null)

  const started = messages.length > 0 || loading // conversation in progress?

  function handleSignedIn(idToken) {
    setToken(idToken)
    sessionStorage.setItem(TOKEN_STORAGE_KEY, idToken)
    // Warm both scale-to-zero services right after sign-in (not before): the GPU VLM and
    // the Layer-3 guard, so their cold starts overlap the user's think-time.
    warmVlm(idToken)
    warmGuard(idToken)
  }

  function handleSignOut() {
    setToken(null)
    sessionStorage.removeItem(TOKEN_STORAGE_KEY)
    window.google?.accounts?.id?.disableAutoSelect()
  }

  // Probe the backend once so we can show a "mock mode" status pill. When there's no
  // login wall (HAS_AUTH false), warm the services on load; with a login wall, warming
  // happens on sign-in instead — no reason to wake billed services for an anon visitor.
  useEffect(() => {
    getHealth()
      .then((h) => {
        setMockBanner(Boolean(h.mock))
        if (!h.mock && !HAS_AUTH) {
          warmVlm()
          warmGuard()
        }
      })
      .catch(() => {}) // health failure is non-fatal for the UI
  }, [])

  // Render the Google "Sign in" button once its script has loaded and we're signed out.
  useEffect(() => {
    if (!HAS_AUTH || token) return
    let cancelled = false
    function tryRender() {
      if (cancelled) return
      if (window.google?.accounts?.id && signinButtonRef.current) {
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: (response) => handleSignedIn(response.credential),
        })
        window.google.accounts.id.renderButton(signinButtonRef.current, {
          theme: 'outline', size: 'large', text: 'signin_with',
        })
      } else {
        setTimeout(tryRender, 100) // GIS script loads async — poll briefly until ready
      }
    }
    tryRender()
    return () => {
      cancelled = true
    }
  }, [token])

  // Revoke object URLs when they change/unmount to avoid leaks.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  // Abort any in-flight request if the component unmounts.
  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  // Keep the newest message in view as the transcript grows.
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  function selectImage(file) {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please choose an image file.')
      return
    }
    if (file.size > MAX_BYTES) {
      setError('Image is larger than 10 MB.')
      return
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setImage(file)
    setPreviewUrl(URL.createObjectURL(file))
    setError('')
  }

  function onFileChange(e) {
    selectImage(e.target.files?.[0])
  }

  function onDrop(e) {
    e.preventDefault()
    selectImage(e.dataTransfer.files?.[0])
  }

  function resetConversation() {
    abortRef.current?.abort()
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setImage(null)
    setPreviewUrl('')
    setQuestion('')
    setMessages([])
    setConversationId(null)
    setError('')
    setStages({})
    setLoading(false)
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setStages({})
    const q = question.trim()
    if (!image) return setError('Please upload an image.')
    if (!q) return setError('Please type a question.')
    if (questionTooWeak(q)) return setError('Please ask a more specific question.')

    // Cancel any in-flight request so a fast resubmit can't race.
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    // Optimistically add the user's turn to the transcript and clear the input.
    setMessages((prev) => [...prev, { role: 'user', text: q }])
    setQuestion('')
    setLoading(true)
    try {
      const result = await askQuestionStream(image, q, {
        signal: controller.signal,
        token,
        conversationId, // null on turn 1 (send image), set on follow-ups (image optional)
        onEvent: (event) => {
          setStages((prev) => ({
            ...prev,
            [event.stage]: { status: event.status, elapsed_ms: event.elapsed_ms },
          }))
        },
      })
      if (result.conversation_id) setConversationId(result.conversation_id)
      if (result.blocked) {
        // Guard (Layer 2/3) or the chart gate rejected the request — show it as an
        // assistant message so the conversation stays coherent.
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', blocked: true, text: result.reason || 'That question was blocked.' },
        ])
        return
      }
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: result.disclaimer || result.answer || '',
          disclaimer: Boolean(result.disclaimer),
          mock: result.mock,
          latency_ms: result.latency_ms,
          detect:
            typeof result.chart_confidence === 'number'
              ? { is_chart: result.is_chart, confidence: result.chart_confidence }
              : null,
          feedback: null, // 'up' | 'down' once the user votes
        },
      ])
    } catch (err) {
      if (err.name === 'AbortError') return // superseded by a newer request
      if (err.authExpired) {
        handleSignOut()
        setError('Your session expired — please sign in again.')
        return
      }
      setError(err.message || 'Something went wrong.')
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setLoading(false)
      }
    }
  }

  async function vote(messageIndex, value) {
    if (!conversationId) return
    // Optimistic: reflect the vote immediately; a failed persist isn't worth interrupting.
    setMessages((prev) =>
      prev.map((m, i) => (i === messageIndex ? { ...m, feedback: value } : m))
    )
    await sendFeedback({ conversationId, vote: value, token })
  }

  const canSubmit = image && question.trim() && !loading

  return (
    <div className="page">
      <nav className="nav-bar">
        <span className="brand">
          <span className="brand-glyph" aria-hidden="true">⚡</span>
          <span className="brand-name">Chart&nbsp;VQA</span>
        </span>
        <span className={`status-pill ${mockBanner ? 'is-mock' : 'is-live'}`}>
          <span className="status-dot" aria-hidden="true" />
          {mockBanner ? 'mock backend' : 'live'}
        </span>
        {started && (
          <button type="button" className="signout" onClick={resetConversation}>
            New chart
          </button>
        )}
        {HAS_AUTH && token && (
          <button type="button" className="signout" onClick={handleSignOut}>
            Sign out
          </button>
        )}
      </nav>

      <main className="container">
        {!started && (
          <header className="hero">
            <p className="eyebrow">CHART QUESTION ANSWERING</p>
            <h1 className="hero-title">Chat with a chart.</h1>
            <p className="hero-sub">
              Upload a chart, then ask follow-up questions about it. Powered by a
              vision-language model behind <code className="chip">POST /api/ask</code>.
            </p>
          </header>
        )}

        {HAS_AUTH && !token ? (
          <div className="card signin-card">
            <p className="signin-prompt">Sign in with Google to start.</p>
            <div ref={signinButtonRef} />
          </div>
        ) : (
          <div className="card chat-card">
            {/* Pinned chart header once a conversation has an image */}
            {previewUrl && (
              <div className="chart-header">
                <img className="chart-thumb" src={previewUrl} alt="Chart in conversation" />
                <div className="chart-meta">
                  <span className="chart-name">{image?.name}</span>
                  <span className="chart-hint">This chart is pinned for the conversation.</span>
                </div>
              </div>
            )}

            {/* Image picker / dropzone — only before an image is chosen */}
            {!previewUrl && (
              <>
                <button
                  type="button"
                  className="picker"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={onDrop}
                  aria-label="Choose a chart image"
                >
                  <span className="picker-empty">
                    <svg className="picker-icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path
                        fill="currentColor"
                        d="M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2M8.5 11l2.5 3 3.5-4.5L19 17H5z"
                      />
                    </svg>
                    <span className="picker-text">Click to choose a chart image</span>
                    <span className="picker-hint">or drag &amp; drop · PNG/JPG · up to 10&nbsp;MB</span>
                  </span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={onFileChange}
                />
              </>
            )}

            {/* Transcript */}
            {messages.length > 0 && (
              <ul className="transcript" aria-live="polite">
                {messages.map((m, i) => (
                  <li key={i} className={`bubble bubble-${m.role}`}>
                    {m.role === 'assistant' ? (
                      <div className={`bubble-body ${m.blocked ? 'bubble-blocked' : ''}`}>
                        {m.detect && !m.detect.is_chart && (
                          <p className="detect detect-warn">
                            ⚠️ This doesn&apos;t look like a chart (
                            {Math.round(m.detect.confidence * 100)}%) — results may be unreliable.
                          </p>
                        )}
                        <div className="markdown">
                          <ReactMarkdown>{m.text}</ReactMarkdown>
                        </div>
                        {!m.blocked && (
                          <div className="bubble-foot">
                            <span className="bubble-meta">
                              {m.mock ? 'mock · ' : ''}
                              {typeof m.latency_ms === 'number'
                                ? `${Number(m.latency_ms).toFixed(0)} ms`
                                : ''}
                            </span>
                            <span className="feedback">
                              <button
                                type="button"
                                className={`vote ${m.feedback === 'up' ? 'vote-on' : ''}`}
                                onClick={() => vote(i, 'up')}
                                aria-label="Good answer"
                                title="Good answer"
                              >
                                👍
                              </button>
                              <button
                                type="button"
                                className={`vote ${m.feedback === 'down' ? 'vote-on' : ''}`}
                                onClick={() => vote(i, 'down')}
                                aria-label="Bad answer"
                                title="Bad answer"
                              >
                                👎
                              </button>
                            </span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="bubble-body">{m.text}</div>
                    )}
                  </li>
                ))}
              </ul>
            )}

            {/* Pending-turn stage progress */}
            {loading && (
              <ul className="stage-list" role="status" aria-live="polite">
                {STAGE_LABELS.filter(({ stage }) => stages[stage]).map(({ stage, label }) => {
                  const s = stages[stage]
                  const done = s.status === 'done'
                  return (
                    <li key={stage} className={`stage-row ${done ? 'stage-done' : ''}`}>
                      {done ? (
                        <span className="stage-check" aria-hidden="true">✓</span>
                      ) : (
                        <span className="spinner stage-spinner" aria-hidden="true" />
                      )}
                      <span className="stage-label">{label}…</span>
                      {done && <span className="stage-time">{Math.round(s.elapsed_ms)} ms</span>}
                    </li>
                  )
                })}
              </ul>
            )}

            {error && (
              <p className="notice" role="alert">
                <span className="notice-dot" aria-hidden="true" />
                {error}
              </p>
            )}

            <div ref={transcriptEndRef} />

            {/* Composer */}
            <form className="composer" onSubmit={onSubmit}>
              <input
                type="text"
                className="text-input"
                placeholder={
                  image
                    ? messages.length
                      ? 'Ask a follow-up…'
                      : 'e.g. What was the revenue in 2024?'
                    : 'Upload a chart image first…'
                }
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={!image || loading}
              />
              <button type="submit" className="submit" disabled={!canSubmit}>
                {loading ? (
                  <span className="submit-loading">
                    <span className="spinner" aria-hidden="true" />
                  </span>
                ) : (
                  'Ask'
                )}
              </button>
            </form>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
