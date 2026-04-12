interface RomanianPlateProps {
  text: string;
  className?: string;
}

export function RomanianPlate({ text, className = "" }: RomanianPlateProps) {
  return (
    <div
      className={`inline-flex items-center justify-center gap-1 rounded border-2 border-blue-600 bg-white px-2 py-1 font-mono font-bold text-black md:px-3 md:py-1.5 ${className}`}
    >
      <div className="flex h-6 w-6 flex-col items-center justify-center rounded-sm bg-blue-600 text-[8px] leading-none text-white md:h-8 md:w-8 md:text-[10px]">
        <span>RO</span>
      </div>
      <span className="text-sm tracking-wider md:text-lg">{text}</span>
    </div>
  );
}
