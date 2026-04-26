interface Props {
  className?: string
  size?: number
}

/**
 * Tesla "T" mark, rendered as inline SVG.
 * Note: Tesla and the Tesla T are trademarks of Tesla, Inc.
 * This site is not affiliated with or endorsed by Tesla.
 */
export default function TeslaT({ className, size = 24 }: Props) {
  return (
    <svg
      viewBox="0 0 40 50"
      width={size}
      height={Math.round(size * 50 / 40)}
      className={className}
      aria-label="Tesla"
      role="img"
      fill="currentColor"
    >
      {/*
        Horizontal top bar with the Tesla characteristic curved arch where
        the stem meets the bar (the two inward curves on either side of center).
        Stem runs down the middle.
      */}
      <path d="
        M20 6
        C20 6 17.5 10.5 12 11.5
        L2 11.5 L2 5 L38 5 L38 11.5
        L28 11.5
        C22.5 10.5 20 6 20 6 Z
        M16.5 11.5 L16.5 47 L23.5 47 L23.5 11.5 Z
      " />
    </svg>
  )
}
