// Chips — 一句可点的话（快速指令 / clarify 选项）。点一下即发送。

interface Props {
  chips: string[]
  disabled?: boolean
  onPick: (chip: string) => void
}

export default function Chips({ chips, disabled, onPick }: Props) {
  if (!chips.length) return null
  return (
    <div className="chat-chips">
      {chips.map((chip) => (
        <button
          key={chip}
          className="chat-chip"
          disabled={disabled}
          onClick={() => onPick(chip)}
        >
          {chip}
        </button>
      ))}
    </div>
  )
}