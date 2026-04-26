export default function Footer() {
  return (
    <footer className="border-t border-line mt-12">
      <div className="max-w-6xl mx-auto px-6 py-8 text-xs muted space-y-2">
        <p className="text-slate-300">
          tslamfm is an educational, open-methodology statistical model. It is <span className="text-bad font-semibold">not</span> investment advice.
        </p>
        <p>
          Live quote pass-through via Yahoo Finance public endpoints. Historical fit and coefficients regenerated on each deploy from a 6-year weekly window using OLS with backward-stable factor selection. See methodology above for what's in and what was rejected.
        </p>
        <p className="text-[11px]">
          © tslamfm · v6-canonical. Built openly. Source available.
        </p>
      </div>
    </footer>
  )
}
