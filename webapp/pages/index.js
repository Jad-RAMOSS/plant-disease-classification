import { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import Head from 'next/head'

function UploadIcon({ className }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  )
}

function ChartIcon({ className }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
      />
    </svg>
  )
}

function WarningIcon({ className }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
      />
    </svg>
  )
}

function LeafIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d="M17 8C8 10 5.9 16.17 3.82 21H5.71C7 18 8.83 16 11 15c2.17-1 4.17-.5 5 0-1 1.17-3 4.5-3 7h2c0-3 1.5-6 4-8 1.17-3.83-.5-8-2-6z"/>
    </svg>
  )
}

const fmt = (cls) =>
  cls
    .replace(/___/g, ' — ')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const isHealthy = (cls) => cls.toLowerCase().includes('healthy')
const isUnidentified = (cls) => cls === 'Unidentified'

function ConfidenceBar({ prediction, index, maxConfidence }) {
  const healthy = isHealthy(prediction.class)
  const isTop = index === 0

  return (
    <div className="group">
      <div className="flex items-center justify-between mb-1 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {isTop && (
            <span className={[
              'text-xs font-bold px-1.5 py-0.5 rounded flex-shrink-0',
              healthy
                ? 'bg-emerald-100 text-emerald-700'
                : 'bg-ecu-red-light text-ecu-red',
            ].join(' ')}>
              #1
            </span>
          )}
          <span className="text-sm text-gray-700 truncate font-medium">
            {fmt(prediction.class)}
          </span>
        </div>
        <span className="text-sm text-gray-400 tabular-nums flex-shrink-0 font-medium">
          {prediction.confidence.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={[
            'h-full rounded-full transition-all duration-700 ease-out',
            isTop
              ? healthy ? 'bg-emerald-500' : 'bg-ecu-red'
              : 'bg-gray-300',
          ].join(' ')}
          style={{ width: `${(prediction.confidence / Math.max(maxConfidence, 1)) * 100}%` }}
        />
      </div>
    </div>
  )
}

export default function Home() {
  const [preview, setPreview] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [serverStatus, setServerStatus] = useState('checking')
  const [classes, setClasses] = useState([])
  const [classesOpen, setClassesOpen] = useState(false)
  const [classSearch, setClassSearch] = useState('')

  useEffect(() => {
    fetch('/api/python/model-info')
      .then((r) => r.json())
      .then((d) => {
        setModelInfo(d)
        setServerStatus('ready')
      })
      .catch(() => setServerStatus('offline'))

    fetch('/api/python/classes')
      .then((r) => r.json())
      .then((d) => setClasses(d.classes || []))
      .catch(() => {})
  }, [])

  const classify = useCallback(async (file) => {
    setLoading(true)
    setError(null)
    setPredictions(null)

    const body = new FormData()
    body.append('image', file)

    try {
      const res = await fetch('/api/python/predict', { method: 'POST', body })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || 'Classification failed')
      } else {
        setPredictions(data.predictions)
      }
    } catch {
      setError('Cannot reach prediction server. Run: python3 predict_server.py')
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback(
    (acceptedFiles) => {
      const file = acceptedFiles[0]
      if (!file) return
      if (preview) URL.revokeObjectURL(preview)
      setPreview(URL.createObjectURL(file))
      setPredictions(null)
      setError(null)
      classify(file)
    },
    [classify, preview]
  )

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] },
    multiple: false,
    noClick: !!preview,
  })

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setPredictions(null)
    setError(null)
  }

  const maxConf = predictions ? predictions[0].confidence : 100
  const topPred = predictions?.[0]

  return (
    <>
      <Head>
        <title>Plant Disease Image Classifier — ECU</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen flex flex-col bg-gray-50 font-sans">
        {/* ── Header ── */}
        <header className="bg-white border-b border-gray-200 shadow-sm">
          <div className="max-w-5xl mx-auto px-6 py-3 flex items-center gap-5">
            <img src="/ecu-logo.png" alt="ECU Logo" className="h-16 w-auto object-contain" />
            <div className="border-l border-gray-200 pl-5">
              <h1 className="text-xl font-bold text-gray-800 tracking-tight leading-tight">
                Plant Disease Image Classifier
              </h1>
              <p className="text-sm text-ecu-gray-light mt-0.5">
                Egyptian Chinese University &mdash; Data Mining Project
              </p>
            </div>

            {/* Server status badge */}
            <div className="ml-auto hidden sm:flex items-center gap-2 text-xs font-medium">
              <span className={[
                'w-2 h-2 rounded-full flex-shrink-0',
                serverStatus === 'ready' ? 'bg-emerald-500'
                  : serverStatus === 'offline' ? 'bg-red-400'
                  : 'bg-yellow-400 animate-pulse',
              ].join(' ')} />
              <span className="text-gray-400">
                {serverStatus === 'ready' ? 'Server ready'
                  : serverStatus === 'offline' ? 'Server offline'
                  : 'Connecting…'}
              </span>
            </div>
          </div>
        </header>

        {/* ── Body ── */}
        <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-8 space-y-6">

          {/* Model info bar */}
          {modelInfo && (
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 bg-white border border-gray-200 rounded-xl px-5 py-3 text-sm text-gray-500">
              <span>
                <span className="font-semibold text-gray-700">Model:</span>{' '}
                {modelInfo.model}
              </span>
              <span className="text-gray-300">|</span>
              <span>
                <span className="font-semibold text-gray-700">Classes:</span>{' '}
                {modelInfo.classes}
              </span>
              <span className="text-gray-300">|</span>
              <span>
                <span className="font-semibold text-gray-700">Input:</span>{' '}
                {modelInfo.img_size?.[0]} × {modelInfo.img_size?.[1]} px
              </span>
            </div>
          )}

          {/* Offline warning */}
          {serverStatus === 'offline' && (
            <div className="flex items-start gap-3 bg-amber-50 border border-amber-200 rounded-xl px-5 py-4 text-sm text-amber-700">
              <WarningIcon className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Prediction server is not running</p>
                <p className="mt-0.5 text-amber-600">
                  Start it with: <code className="bg-amber-100 px-1.5 py-0.5 rounded font-mono text-xs">python3 predict_server.py</code>
                </p>
              </div>
            </div>
          )}

          {/* Main two-column layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* ── Upload panel ── */}
            <div className="flex flex-col gap-3">
              <h2 className="text-xs font-semibold uppercase tracking-widest text-ecu-gray-light">
                Input Image
              </h2>

              <div
                {...getRootProps()}
                className={[
                  'relative rounded-2xl border-2 border-dashed cursor-pointer',
                  'transition-all duration-200 min-h-72 flex items-center justify-center overflow-hidden',
                  isDragActive
                    ? 'border-ecu-red bg-ecu-red-light scale-[0.99] shadow-inner'
                    : preview
                    ? 'border-gray-200 bg-white hover:border-ecu-red/50'
                    : 'border-gray-200 bg-white hover:border-ecu-red hover:bg-ecu-red-light/40',
                ].join(' ')}
              >
                <input {...getInputProps()} />

                {preview ? (
                  <>
                    <img
                      src={preview}
                      alt="Uploaded plant"
                      className="w-full h-72 object-contain"
                    />
                    {isDragActive && (
                      <div className="absolute inset-0 bg-ecu-red/10 flex items-center justify-center rounded-2xl">
                        <p className="font-semibold text-ecu-red">Drop to replace</p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center p-10 select-none">
                    <div className="w-16 h-16 rounded-2xl bg-ecu-red-light flex items-center justify-center mx-auto mb-4 border border-ecu-red/20">
                      <UploadIcon className="w-8 h-8 text-ecu-red" />
                    </div>
                    <p className="font-semibold text-gray-700 text-base">
                      {isDragActive ? 'Drop the image here' : 'Drag & drop a plant image'}
                    </p>
                    <p className="text-gray-400 text-sm mt-1.5">or click to browse files</p>
                    <p className="text-gray-300 text-xs mt-5">
                      JPG &bull; PNG &bull; WebP &bull; BMP
                    </p>
                  </div>
                )}
              </div>

              {preview && (
                <div className="flex items-center gap-4">
                  <button
                    onClick={clear}
                    className="text-sm text-gray-400 hover:text-ecu-red transition-colors"
                  >
                    ✕ Clear image
                  </button>
                  <button
                    onClick={open}
                    className="text-sm text-gray-400 hover:text-ecu-red transition-colors"
                  >
                    ↑ Upload new
                  </button>
                </div>
              )}
            </div>

            {/* ── Results panel ── */}
            <div className="flex flex-col gap-3">
              <h2 className="text-xs font-semibold uppercase tracking-widest text-ecu-gray-light">
                Classification Results
              </h2>

              <div className="flex-1 bg-white rounded-2xl border border-gray-200 min-h-72 p-6 flex flex-col">

                {/* Loading */}
                {loading && (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4">
                    <div className="relative w-12 h-12">
                      <div className="absolute inset-0 rounded-full border-4 border-ecu-red-light" />
                      <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-ecu-red animate-spin" />
                    </div>
                    <p className="text-sm text-gray-400">Analyzing image&hellip;</p>
                  </div>
                )}

                {/* Error */}
                {error && !loading && (
                  <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center px-4">
                    <div className="w-12 h-12 rounded-full bg-ecu-red-light flex items-center justify-center">
                      <WarningIcon className="w-6 h-6 text-ecu-red" />
                    </div>
                    <p className="font-semibold text-gray-700">Classification failed</p>
                    <p className="text-sm text-gray-400 leading-relaxed">{error}</p>
                  </div>
                )}

                {/* Empty state */}
                {!loading && !error && !predictions && (
                  <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center">
                    <ChartIcon className="w-12 h-12 text-gray-200" />
                    <p className="text-sm text-gray-400">
                      Upload a plant image to see predictions
                    </p>
                  </div>
                )}

                {/* Predictions */}
                {predictions && !loading && (
                  <div className="flex flex-col gap-5 flex-1">

                    {/* Top prediction banner */}
                    <div className={[
                      'rounded-xl p-4 border',
                      isUnidentified(topPred.class)
                        ? 'bg-amber-50 border-amber-300'
                        : isHealthy(topPred.class)
                        ? 'bg-emerald-50 border-emerald-200'
                        : 'bg-ecu-red-light border-ecu-red/20',
                    ].join(' ')}>
                      <div className="flex items-start gap-3">
                        <div className={[
                          'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
                          isUnidentified(topPred.class)
                            ? 'bg-amber-100'
                            : isHealthy(topPred.class)
                            ? 'bg-emerald-100'
                            : 'bg-ecu-red/10',
                        ].join(' ')}>
                          {isUnidentified(topPred.class)
                            ? <WarningIcon className="w-5 h-5 text-amber-500" />
                            : <LeafIcon className={[
                                'w-5 h-5',
                                isHealthy(topPred.class) ? 'text-emerald-600' : 'text-ecu-red',
                              ].join(' ')} />
                          }
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-0.5">
                            Prediction
                          </p>
                          <p className={[
                            'font-bold text-base leading-tight',
                            isUnidentified(topPred.class)
                              ? 'text-amber-700'
                              : isHealthy(topPred.class)
                              ? 'text-emerald-700'
                              : 'text-ecu-red',
                          ].join(' ')}>
                            {isUnidentified(topPred.class) ? 'Unidentified' : fmt(topPred.class)}
                          </p>
                          {isUnidentified(topPred.class) ? (
                            <p className="text-sm text-amber-600 mt-1 leading-snug">
                              {topPred.message}
                            </p>
                          ) : (
                            <p className={[
                              'text-sm font-semibold mt-1',
                              isHealthy(topPred.class) ? 'text-emerald-600' : 'text-ecu-red/80',
                            ].join(' ')}>
                              {topPred.confidence.toFixed(2)}% confidence
                            </p>
                          )}
                        </div>
                        <span className={[
                          'text-xs font-bold px-2 py-1 rounded-full flex-shrink-0',
                          isUnidentified(topPred.class)
                            ? 'bg-amber-100 text-amber-700'
                            : isHealthy(topPred.class)
                            ? 'bg-emerald-100 text-emerald-700'
                            : 'bg-ecu-red/10 text-ecu-red',
                        ].join(' ')}>
                          {isUnidentified(topPred.class)
                            ? 'Unknown'
                            : isHealthy(topPred.class) ? 'Healthy' : 'Diseased'}
                        </span>
                      </div>
                    </div>

                    {/* Top 5 bars — hidden for unidentified */}
                    {!isUnidentified(topPred.class) && (
                      <div className="flex flex-col gap-3.5 flex-1">
                        <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                          Top 5 Candidates
                        </p>
                        {predictions.map((p, i) => (
                          <ConfidenceBar
                            key={i}
                            prediction={p}
                            index={i}
                            maxConfidence={maxConf}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Classes dropdown */}
          {classes.length > 0 && (
            <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden">
              {/* Header / toggle */}
              <button
                onClick={() => setClassesOpen((o) => !o)}
                className="w-full flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors text-left"
              >
                <div className="flex items-center gap-3">
                  <span className="text-xs font-semibold uppercase tracking-widest text-ecu-gray-light">
                    Supported Disease Classes
                  </span>
                  <span className="bg-ecu-red-light text-ecu-red text-xs font-bold px-2 py-0.5 rounded-full">
                    {classes.length}
                  </span>
                </div>
                <svg
                  className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${classesOpen ? 'rotate-180' : ''}`}
                  fill="none" stroke="currentColor" viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Collapsible body */}
              {classesOpen && (
                <div className="border-t border-gray-100 px-6 pb-6 pt-4">
                  {/* Search */}
                  <div className="relative mb-4">
                    <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-4.35-4.35M17 11A6 6 0 105 11a6 6 0 0012 0z" />
                    </svg>
                    <input
                      type="text"
                      placeholder="Search classes…"
                      value={classSearch}
                      onChange={(e) => setClassSearch(e.target.value)}
                      className="w-full pl-9 pr-4 py-2 text-sm border border-gray-200 rounded-xl bg-gray-50 focus:outline-none focus:ring-2 focus:ring-ecu-red/30 focus:border-ecu-red/50 placeholder-gray-300"
                    />
                    {classSearch && (
                      <button
                        onClick={() => setClassSearch('')}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-300 hover:text-gray-500"
                      >
                        ✕
                      </button>
                    )}
                  </div>

                  {/* Grid of class pills */}
                  {(() => {
                    const filtered = classes.filter((c) =>
                      fmt(c.name).toLowerCase().includes(classSearch.toLowerCase())
                    )

                    // Group by plant (everything before ___)
                    const grouped = filtered.reduce((acc, c) => {
                      const plant = c.name.split('___')[0].replace(/_/g, ' ').replace(/[()]/g, '').trim()
                      if (!acc[plant]) acc[plant] = []
                      acc[plant].push(c)
                      return acc
                    }, {})

                    if (filtered.length === 0) {
                      return (
                        <p className="text-sm text-gray-400 text-center py-4">
                          No classes match &ldquo;{classSearch}&rdquo;
                        </p>
                      )
                    }

                    return (
                      <div className="space-y-4">
                        {Object.entries(grouped).map(([plant, items]) => (
                          <div key={plant}>
                            <p className="text-xs font-semibold uppercase tracking-wider text-ecu-gray-light mb-2">
                              {plant}
                            </p>
                            <div className="flex flex-wrap gap-2">
                              {items.map((c) => {
                                const disease = c.name.split('___')[1]?.replace(/_/g, ' ').trim() || 'healthy'
                                const healthy = isHealthy(c.name)
                                return (
                                  <span
                                    key={c.index}
                                    className={[
                                      'inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full border',
                                      healthy
                                        ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                                        : 'bg-gray-50 border-gray-200 text-gray-600',
                                    ].join(' ')}
                                  >
                                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${healthy ? 'bg-emerald-400' : 'bg-ecu-red'}`} />
                                    {disease}
                                  </span>
                                )
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    )
                  })()}
                </div>
              )}
            </div>
          )}

          {/* About section */}
          <div className="bg-white rounded-2xl border border-gray-200 px-6 py-5">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-ecu-gray-light mb-3">
              About
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
              {[
                { label: 'Architecture', value: 'EfficientNetB3' },
                { label: 'Dataset', value: 'PlantVillage' },
                { label: 'Classes', value: '38' },
                { label: 'Plant Species', value: '14' },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-50 rounded-xl px-4 py-3">
                  <p className="text-lg font-bold text-gray-800">{value}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{label}</p>
                </div>
              ))}
            </div>
          </div>
        </main>

        {/* ── Footer ── */}
        <footer className="border-t border-gray-200 bg-white py-4 mt-auto">
          <p className="text-center text-xs text-gray-300">
            Egyptian Chinese University &nbsp;&bull;&nbsp; Data Mining Project &nbsp;&bull;&nbsp; EfficientNetB3 &nbsp;&bull;&nbsp; PlantVillage
          </p>
        </footer>
      </div>
    </>
  )
}
