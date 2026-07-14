import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { askQuestionStream, clearConversation, getConversation, getHealth, warmVlm, warmGuard, sendFeedback } from './api'
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
  // True once the "vlm" stage has been running a while — the VLM is a scale-to-zero Cloud
  // Run GPU service, so the FIRST request after it's been idle pays a real cold-start cost
  // (~90-100s to load the model) hidden inside this one stage. Surfaced as a hint, not
  // measured precisely — see the timer effect below.
  const [coldStartHint, setColdStartHint] = useState(false)
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

  // Fetch the signed-in user's own conversation from the backend and populate the
  // transcript with it (Phase 5.1). Backend-authoritative and per-user (derived from the
  // token server-side, see app.py's GET /api/conversation) — this is what lets each
  // account keep its own history without ever inheriting another account's chat.
  async function restoreConversation(idToken) {
    const restored = await getConversation(idToken)
    if (restored?.conversation_id && restored.messages?.length) {
      setConversationId(restored.conversation_id)
      // The backend sends snake_case (image_data_uri, image_index); a live turn instead
      // sets imageUrl/imageIndex (see onSubmit) — normalize restored messages to the same
      // shape so the transcript renders chart bubbles + the "Image N" caption identically.
      setMessages(
        restored.messages.map((m) => ({
          ...m,
          ...(m.image_data_uri && { imageUrl: m.image_data_uri }),
          ...(m.image_index && { imageIndex: m.image_index }),
        }))
      )
    }
  }

  function handleSignedIn(idToken) {
    // Defense in depth alongside handleSignOut's reset: a second account signing in on the
    // same tab (Google's "choose an account" flow, or a silently-replaced expired token)
    // must never inherit a still-in-memory chat from whoever was signed in before —
    // restoreConversation() below then repopulates THIS account's own history, if any.
    if (token && token !== idToken) resetConversation()
    setToken(idToken)
    sessionStorage.setItem(TOKEN_STORAGE_KEY, idToken)
    // Warm both scale-to-zero services right after sign-in (not before): the GPU VLM and
    // the Layer-3 guard, so their cold starts overlap the user's think-time.
    warmVlm(idToken)
    warmGuard(idToken)
    restoreConversation(idToken)
  }

  function handleSignOut() {
    setToken(null)
    sessionStorage.removeItem(TOKEN_STORAGE_KEY)
    window.google?.accounts?.id?.disableAutoSelect()
    // Clears the VISIBLE transcript only — the conversation itself is still persisted
    // backend-side under this user's hashed id (conversation_store's TTL) and comes back
    // via restoreConversation() next time they sign in. Without this reset, the next
    // person to sign in on the same tab would see the previous user's chat before their
    // own restore call resolves.
    resetConversation()
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

  // If the page loads already signed in (token restored from sessionStorage, App.jsx:48-50
  // — this path never calls handleSignedIn), restore this user's conversation the same way
  // a fresh sign-in does. Runs once: token here is the token useState was initialized with.
  useEffect(() => {
    if (token) restoreConversation(token)
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  // NOTE: object URLs are NOT auto-revoked when previewUrl/messages change, because once
  // an image is sent its URL moves into a message and must stay valid to render in the
  // transcript. We track every URL we mint in a ref and revoke them all only on unmount
  // (and per-URL in clearImage/resetConversation). The PENDING preview is revoked
  // explicitly in selectImage/clearImage.
  const objectUrlsRef = useRef(new Set())

  // Abort any in-flight request + revoke every minted object URL, only on true unmount.
  useEffect(() => {
    const urls = objectUrlsRef.current // stable Set, created once — snapshot for cleanup
    const abort = abortRef
    return () => {
      abort.current?.abort()
      urls.forEach((u) => URL.revokeObjectURL(u))
      urls.clear()
    }
  }, [])

  // Keep the newest message in view as the transcript grows — including mid-turn, as
  // per-stage SSE progress rows (chart_gate/guard/vlm) or streamed answer text arrive.
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading, stages])

  // Cold-start hint: the "vlm" stage covers both a warm answer (~1-2s) and a cold Cloud
  // Run GPU spin-up (~90-100s to load the model) with no distinct signal between them —
  // so after a few seconds of "vlm" still running, assume it's cold and say so, rather
  // than leaving the user staring at a spinner with no idea why it's slow.
  useEffect(() => {
    if (stages.vlm?.status !== 'start') {
      setColdStartHint(false)
      return
    }
    const t = setTimeout(() => setColdStartHint(true), 8000)
    return () => clearTimeout(t)
  }, [stages.vlm?.status])

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
    // Revoke the previous PENDING preview (not yet sent, so not owned by a message).
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      objectUrlsRef.current.delete(previewUrl)
    }
    const url = URL.createObjectURL(file)
    objectUrlsRef.current.add(url)
    setImage(file)
    setPreviewUrl(url)
    setError('')
  }

  function onFileChange(e) {
    selectImage(e.target.files?.[0])
  }

  // Clear the chosen-but-not-yet-sent image so the user can pick a different one.
  function clearImage() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      objectUrlsRef.current.delete(previewUrl)
    }
    setImage(null)
    setPreviewUrl('')
    setError('')
    if (fileInputRef.current) fileInputRef.current.value = '' // allow re-picking the same file
  }

  function onDrop(e) {
    e.preventDefault()
    selectImage(e.dataTransfer.files?.[0])
  }

  function resetConversation() {
    abortRef.current?.abort()
    // Revoke every URL minted this session (pending preview + all message-owned images).
    objectUrlsRef.current.forEach((u) => URL.revokeObjectURL(u))
    objectUrlsRef.current.clear()
    setImage(null)
    setPreviewUrl('')
    setQuestion('')
    setMessages([])
    setConversationId(null)
    setError('')
    setStages({})
    setLoading(false)
  }

  // The "New session" button: clears the visible chat AND tells the backend to forget
  // this user's current conversation, so it does NOT come back via restoreConversation()
  // on a future sign-in/reload. Distinct from resetConversation() alone, which is also
  // used by sign-out/sign-in (where the conversation SHOULD still be restorable later).
  function startNewSession() {
    resetConversation()
    if (token) clearConversation(token)
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setStages({})
    const q = question.trim()
    const inConversation = messages.length > 0
    // An image is required to START a conversation; on a follow-up it's optional (the
    // server reuses the conversation's stored charts), but the user MAY attach a new one.
    if (!inConversation && !image) return setError('Please upload an image.')
    if (!q) return setError('Please type a question.')
    if (questionTooWeak(q)) return setError('Please ask a more specific question.')

    // Cancel any in-flight request so a fast resubmit can't race.
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    // Optimistically add the user's turn to the transcript. If an image is attached this
    // turn (turn 1, or a mid-conversation upload), it rides on THIS user bubble
    // (ChatGPT-style: image above the question that introduced it). Clear the input +
    // the just-consumed image attachment.
    const attachedUrl = image ? previewUrl : null
    const messageIndex = messages.length
    setMessages((prev) => [...prev, { role: 'user', text: q, imageUrl: attachedUrl }])
    setQuestion('')
    // Detach the image now that it's committed to a message (its object URL stays valid —
    // it's referenced by the message; a new pick / reset revokes it).
    setImage(null)
    setPreviewUrl('')
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
      // Tag the user bubble with the image number the server assigned, so the transcript
      // can show "Image N" and the user knows how to reference it ("ask about image 1").
      if (result.image_index) {
        setMessages((prev) =>
          prev.map((m, i) => (i === messageIndex ? { ...m, imageIndex: result.image_index } : m))
        )
      }
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

  // Can submit when there's a question AND either an image is attached OR the conversation
  // already has charts (a follow-up reuses the stored images).
  const canSubmit = (image || started) && question.trim() && !loading

  return (
    <div className="page">
      <nav className="nav-bar">
        <span className="brand">
          <svg className="brand-glyph" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <rect x="4" y="13" width="4" height="7" rx="0.8" fill="currentColor" />
            <rect x="10" y="9" width="4" height="11" rx="0.8" fill="currentColor" />
            <rect x="16" y="4" width="4" height="16" rx="0.8" fill="currentColor" />
          </svg>
          <span className="brand-name">Chartbot&nbsp;QA</span>
        </span>
        <span className={`status-pill ${mockBanner ? 'is-mock' : 'is-live'}`}>
          <span className="status-dot" aria-hidden="true" />
          {mockBanner ? 'mock backend' : 'live'}
        </span>
        {!started && (
          <p className="hero-sub">Upload a chart, then keep asking — follow-up questions,
            the same conversation.</p>
        )}
        {started && (
          <button type="button" className="signout" onClick={startNewSession}>
            New session
          </button>
        )}
        {HAS_AUTH && token && (
          <button type="button" className="signout" onClick={handleSignOut}>
            Sign out
          </button>
        )}
      </nav>

      <main className="container">
        {HAS_AUTH && !token ? (
          <div className="card signin-card">
            <p className="signin-prompt">Sign in with Google to start.</p>
            <div ref={signinButtonRef} />
          </div>
        ) : (
          <div className="card chat-card">
            {/* Selected-chart confirmation — compact, only BEFORE the first question is
                sent. Once sent, the image moves into the first user bubble (ChatGPT-style),
                so this chip disappears and doesn't duplicate it. */}
            {previewUrl && messages.length === 0 && (
              <div className="chart-header">
                <img className="chart-thumb" src={previewUrl} alt="Selected chart" />
                <div className="chart-meta">
                  <span className="chart-name">{image?.name}</span>
                  <span className="chart-hint">Ask a question about this chart below.</span>
                </div>
                <button
                  type="button"
                  className="chart-remove"
                  onClick={clearImage}
                  aria-label="Remove image"
                  title="Remove image"
                >
                  ✕
                </button>
              </div>
            )}

            {/* Image picker / dropzone — turn-1 only. Once the conversation has started,
                attaching a NEW image goes through the composer's attach button instead
                (this big dropzone would otherwise reappear whenever previewUrl is cleared
                after each send, which it is — the sent image lives in the message now). */}
            {!started && !previewUrl && (
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

            {/* Transcript. When empty (no messages yet), an inert flex spacer takes its
                place so the composer below still gets pushed to the bottom of the card
                instead of sitting right under the chart-header/picker. */}
            {messages.length === 0 && <div className="transcript-spacer" aria-hidden="true" />}
            {messages.length > 0 && (
              <ul className="transcript" aria-live="polite">
                {messages.map((m, i) => (
                  <li key={i} className={`bubble bubble-${m.role}`}>
                    {m.role === 'assistant' ? (
                      <div className={`bubble-body ${m.blocked ? 'bubble-blocked' : ''}`}>
                        {m.detect && !m.detect.is_chart && (
                          <p className="detect detect-warn">
                            ⚠️ This may not be a chart — the answer might be unreliable.
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
                      <div className="bubble-user-wrap">
                        {m.imageUrl && (
                          <figure className="bubble-figure">
                            <img className="bubble-image" src={m.imageUrl} alt="Uploaded chart" />
                            {m.imageIndex && (
                              <figcaption className="bubble-image-cap">
                                Image {m.imageIndex}
                              </figcaption>
                            )}
                          </figure>
                        )}
                        <div className="bubble-body">{m.text}</div>
                      </div>
                    )}
                  </li>
                ))}
                <li ref={transcriptEndRef} className="transcript-end" aria-hidden="true" />
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

            {loading && coldStartHint && (
              <p className="notice notice-info" role="status">
                <span className="notice-dot" aria-hidden="true" />
                First answer after a while can take a bit longer — the model is starting up.
              </p>
            )}

            {error && (
              <p className="notice" role="alert">
                <span className="notice-dot" aria-hidden="true" />
                {error}
              </p>
            )}

            {/* Pending mid-conversation attachment (a chart added for the NEXT question) */}
            {started && previewUrl && (
              <div className="attach-chip">
                <img className="attach-thumb" src={previewUrl} alt="Attached chart" />
                <span className="attach-name">{image?.name || 'New chart'}</span>
                <button
                  type="button"
                  className="attach-remove"
                  onClick={clearImage}
                  aria-label="Remove attachment"
                  title="Remove attachment"
                >
                  ✕
                </button>
              </div>
            )}

            {/* Composer */}
            <form className="composer" onSubmit={onSubmit}>
              {/* Attach a chart to the next question (ChatGPT-style). Only mid-conversation;
                  turn 1 uses the big picker above. */}
              {started && (
                <button
                  type="button"
                  className="attach-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  aria-label="Attach a chart image"
                  title="Attach a chart image"
                >
                  <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
                    <path
                      fill="currentColor"
                      d="M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2M8.5 11l2.5 3 3.5-4.5L19 17H5z"
                    />
                  </svg>
                </button>
              )}
              <input
                type="text"
                className="text-input"
                placeholder={
                  started
                    ? 'Ask a follow-up…'
                    : image
                      ? 'e.g. What was the revenue in 2024?'
                      : 'Upload a chart image first…'
                }
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={(!image && !started) || loading}
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
