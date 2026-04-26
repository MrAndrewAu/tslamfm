interface Props {
  modelVersion: string
  generatedAt: string
}

export default function Header({ modelVersion, generatedAt }: Props) {
  const date = new Date(generatedAt).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
  })
  return (
    <header className="border-b border-line">
      <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-accent/15 border border-accent/30 flex items-center justify-center font-bold text-accent">T</div>
          <div>
            <div className="text-lg font-semibold tracking-tight">tslamfm</div>
            <div className="text-xs muted -mt-0.5">TSLA multi-factor model</div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs muted">model</div>
          <div className="text-sm mono">{modelVersion}</div>
          <div className="text-[10px] muted">data: {date}</div>
        </div>
      </div>
    </header>
  )
}
